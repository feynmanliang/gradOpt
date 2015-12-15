package com.feynmanliang.optala

import breeze.linalg._
import breeze.numerics.ceil
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister

import scala.util.Random

case class Generation(population: Seq[(Vector[Double],Double)]) {
  def meanNegFitness(): Double = population.map(_._2).sum / (1D*population.size)
  def bestIndividual(): (Vector[Double],Double) = population.minBy(_._2)
}

class GeneticAlgorithm(
    var maxObjectiveEvals: Int = Int.MaxValue,
    var maxSteps: Int = 1000) {
  /**
  *  Minimizes `f` subject to decision variables inside hypercube defined by `lb` and `ub`.
  **/
  def minimize(
      f: Vector[Double] => Double,
      lb: Vector[Double],
      ub: Vector[Double],
      popSize: Int = 20,
      selectionStrategy: SelectionStrategy = FitnessProportionateSelection,
      eliteCount: Int = 2,
      xoverFrac: Double = 0.8,
      seed: Option[Long] = None): (Option[Vector[Double]], Option[PerfDiagnostics[Generation]]) = {
    implicit val randBasis: RandBasis = seed match {
      case Some(s) =>
        Random.setSeed(s)
        new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(s)))
      case None => new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister()))
    }
    val fCnt = new FunctionWithCounter(f)

    val xoverCount: Int = ((popSize - eliteCount) * xoverFrac).ceil.toInt
    val mutantCount: Int = popSize - eliteCount - xoverCount
    val initGen = initialize(fCnt, lb, ub, popSize)

    val successors: Stream[Generation] = Stream.iterate(initGen) { gen =>
      val numParents = ((max(2*xoverCount, mutantCount)+1) / 2) * 2 // ensure even pairs for xover (TODO: improve)
      val parents = selectParents(gen.population, selectionStrategy, numParents)

      val elites = gen.population.sortBy(_._2).take(eliteCount)
      val xovers = crossOver(fCnt, parents)
      val mutants = mutate(fCnt, lb, ub, parents, mutantCount)

      Generation(elites ++ xovers ++ mutants)
    }

    val iters = successors
      .take(maxSteps)
      .takeWhile(_ => fCnt.numCalls <= maxObjectiveEvals)

    // implements "best solution archiving" by finding best solution within lazy stream up until termination
    val bestSolution = iters.minBy(_.population.map(_._2).min)
    val perf = PerfDiagnostics(
      iters.toList,
      fCnt.numCalls,
      0
    )
    (Some(iters.last.population.minBy(_._2)._1), Some(perf))
  }

  private[optala] def initialize(
      f: Vector[Double] => Double, lb: Vector[Double], ub: Vector[Double], popSize: Int)(
      implicit randBasis: RandBasis = Rand): Generation = {
    val n = lb.size
    val center = (ub + lb) / 2D
    val range = ub - lb
    Generation(List.fill(popSize) {
      val x = (range :* DenseVector.rand(n, Uniform(-.5D,.5D))) + center
      (x, f(x))
    })
  }

  private[optala] def selectParents(
      pop: Seq[(Vector[Double],Double)],
      strategy: SelectionStrategy,
      n: Int)(
      implicit randBasis: RandBasis = Rand): Seq[(Vector[Double],Double)] = strategy match {
    case FitnessProportionateSelection =>
      val minFitness = -1D*pop.map(_._2).max
      // TODO: fix multinomial random seed
      val dist = new Multinomial(Counter(pop.map(x => (x, -1D*x._2 - minFitness + 1D))))
      dist.sample(n)
    case StochasticUniversalSampling =>
      val minFitness = -1D*pop.map(_._2).max
      // (individual, fitness normalized s.t. > 0)
      val popNorm = pop.map { case (individual, negFitness) =>
        val fitness =  -1D*negFitness - minFitness + 1D
        (individual, fitness)
      }
      val f = popNorm.map(_._2).sum
      val stepSize = f / n

      // (point, negative fitness, sum of fitnesses normalized to be above zero)
      val popWithCumSums = popNorm.foldRight(List[(Vector[Double],Double,Double)]()) { case ((individual, fitness),acc) =>
        acc match {
          case Nil => (individual, fitness, fitness) :: acc
          case (_,_,cumFitness)::_ => (individual, fitness, fitness + cumFitness) :: acc
        }
      }.reverse

      val start = new Uniform(0, stepSize).sample()
      (start to n*stepSize by stepSize).map { p =>
        val (point, negFit, _) = popWithCumSums.dropWhile(_._3 < p).head
        (point, negFit)
      }
    case TournamentSelection(tournamentProp) =>
      require(
        0D <= tournamentProp && tournamentProp <= 1D,
        s"tournament proportion must be in [0,1] but got $tournamentProp")
      Seq.fill(n) {
        Random.shuffle(pop)
          .take(ceil(tournamentProp*pop.size).toInt)
          .minBy(_._2) // min by negative fitness = objective value
      }
  }

  private[optala] def crossOver(
      f: Vector[Double] => Double,
      parents: Seq[(Vector[Double],Double)])(
      implicit randBasis: RandBasis = Rand): Seq[(Vector[Double],Double)] = {
    val dist = Uniform(0D, 1D)
    parents.grouped(2).map { x =>
      val p1 = x.head._1
      val p2 = x(1)._1
      val t = dist.sample()
      val child = t*p1 + (1D - t)*p2
      (child, f(child))
    }.toSeq
  }

  private[optala] def mutate(
      f: Vector[Double] => Double,
      lb: Vector[Double],
      ub: Vector[Double],
      parents: Seq[(Vector[Double],Double)],
      mutantCount: Int)(
      implicit randBasis: RandBasis = Rand): Seq[(Vector[Double],Double)] = {
    Random.shuffle(parents).take(mutantCount).map { x =>
      val delta = DenseVector.rand[Double](x._1.size, Gaussian(0, 1))
      val child = max(min(x._1 + delta, ub), lb)
      (child, f(child))
    }
  }
}

sealed trait SelectionStrategy
case object FitnessProportionateSelection extends SelectionStrategy
case object StochasticUniversalSampling extends SelectionStrategy
case class TournamentSelection(tournamentProportion: Double) extends SelectionStrategy
