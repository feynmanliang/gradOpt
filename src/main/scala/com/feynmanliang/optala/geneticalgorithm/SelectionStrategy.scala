package com.feynmanliang.optala.geneticalgorithm

import breeze.linalg.Counter
import breeze.numerics.ceil
import breeze.stats.distributions.{Multinomial, RandBasis, Uniform}

import scala.util.Random

sealed trait SelectionStrategy {
  /** Selects parents for the next generation.
    * @param pop population to choose parents from
    * @param numParents number of parents to select
    * @param randBasis optional random seed, randomly initialized if omitted
    */
  private[geneticalgorithm] def selectParents(
    pop: Seq[Individual], numParents: Int)(implicit randBasis: RandBasis): Seq[Individual]
}

/** Selects parents by drawing from a categorical with probabilities proportional to fitness. */
case object FitnessProportionateSelection extends SelectionStrategy {
  override def selectParents(pop: Seq[Individual], numParents: Int)(
      implicit randBasis: RandBasis): Seq[Individual] = {
    val minFitness = -1D*pop.map(_.objVal).max
    Multinomial(Counter(pop.map(x => (x, -1D*x.objVal - minFitness + 1D)))).sample(numParents)
  }
}

/** Selects parents by choosing a single random value and sampling parents at uniformly spaced intervals.
  * @see {https://en.wikipedia.org/wiki/Stochastic_universal_sampling}
  */
case object StochasticUniversalSampling extends SelectionStrategy {
  override def selectParents(pop: Seq[Individual], numParents: Int)(
      implicit randBasis: RandBasis): Seq[Individual] = {
    val minFitness = pop.map(_.fitness).min
    // (individual, fitness normalized s.t. > 0)
    val popNorm = pop.map { case individual =>
      val normFitness = individual.fitness - minFitness + 1D
      (individual, normFitness)
    }
    val f = popNorm.map(_._2).sum
    val stepSize = f / numParents
    val popNormSortedWithCumSums = popNorm
      .sortBy(-1D * _._2) // sort by descending fitness
      .foldRight(List[(Individual, Double)]()) {
      // calculate cumulative sums of normalized fitness
      case ((individual, normFit), acc) =>
        acc match {
          case Nil => (individual, normFit) :: acc
          case (_, cumFitness) :: _ => (individual, normFit + cumFitness) :: acc
        }
    }
      .reverse
    val start = new Uniform(0, stepSize)(randBasis).sample()
    (start to numParents * stepSize by stepSize).map { p =>
      val (individual, _) = popNormSortedWithCumSums.dropWhile(_._2 < p).head
      individual
    }
  }
}

/** Selects parents by taking random `tournamentProportion` proportion subsets of the population and selecting the best
  * individual from each subset.
  */
case class TournamentSelection(tournamentProportion: Double) extends SelectionStrategy {
  override def selectParents(pop: Seq[Individual], numParents: Int)(
      implicit randBasis: RandBasis): Seq[Individual] = {
    require(
      0D <= tournamentProportion && tournamentProportion <= 1D,
      s"tournament proportion must be in [0,1] but got $tournamentProportion")
    Seq.fill(numParents) {
      Random.shuffle(pop)
        .take(ceil(tournamentProportion*pop.size).toInt)
        .maxBy(_.fitness)
    }
  }
}

