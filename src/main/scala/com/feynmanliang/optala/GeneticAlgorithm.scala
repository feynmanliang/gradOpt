package com.feynmanliang.optala

import breeze.numerics.ceil
import org.apache.commons.math3.random.MersenneTwister

import scala.util.Random

import breeze.linalg._
import breeze.stats.distributions._

private[optala] case class Individual(
    override val f: Vector[Double] => Double,
    override val point: DenseVector[Double]) extends Solution(f, point) {
  val fitness = -1D * objVal
}

private[optala] case class Generation(population: Seq[Individual]) {
  def averageObjVal: Double = population.map(_.objVal).sum / population.size
  def averageFitness: Double = -1D * averageObjVal
  val bestPoint = population.minBy(_.objVal)
}

private[optala] case class GARunResult(
  override val stateTrace: List[Generation],
  override val numObjEval: Long,
  override val numGradEval: Long) extends RunResult[Generation] {
  override val bestSolution = stateTrace.maxBy(_.bestPoint.objVal).bestPoint
}

/** An implementation of the Genetic Algorithm (GA) optimization method.
  *
  * @param maxObjectiveEvals maximum number of objective function evaluations before termination
  * @param maxIter maximum number of iterations before termination
  */
class GeneticAlgorithm(
    var maxObjectiveEvals: Int = Int.MaxValue,
    var maxIter: Int = 1000) {

  /** Minimizes `f` subject to decision variables inside hypercube defined by `lb` and `ub`.
    *
    * @param f objective function
    * @param lb lower bounds of hypercube
    * @param ub upper bounds of hypercube
    * @param popSize population size
    * @param selectionStrategy selection strategy
    * @param eliteCount elite count
    * @param xoverFrac crossover fraction
    * @param seed random seed
    */
  def minimize(
      f: Vector[Double] => Double,
      lb: DenseVector[Double],
      ub: DenseVector[Double],
      popSize: Int = 20,
      selectionStrategy: SelectionStrategy = FitnessProportionateSelection,
      eliteCount: Int = 2,
      xoverFrac: Double = 0.8,
      seed: Option[Long] = None): RunResult[Generation] = {
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
      val numParents = ((max(2*xoverCount, mutantCount)+1) / 2) * 2 // ensure even pairs for crossover
      val parents = selectParents(gen.population, selectionStrategy, numParents)

      val elites = gen.population.sortBy(_.objVal).take(eliteCount)
      val xovers = crossOver(fCnt, parents)
      val mutants = mutate(fCnt, lb, ub, parents, mutantCount)

      Generation(elites ++ xovers ++ mutants)
    }

    val trace = successors
      .takeWhile(_ => fCnt.numCalls <= maxObjectiveEvals)
      .take(maxIter)
      .toList
    GARunResult(trace, fCnt.numCalls, 0)
  }

  /** Initializes a population to randomly lie within the hypercube given by `lb` and `ub` */
  private[optala] def initialize(
      f: Vector[Double] => Double, lb: DenseVector[Double], ub: DenseVector[Double], popSize: Int)(
      implicit randBasis: RandBasis = Rand): Generation = {
    val n = lb.size
    val center = (ub + lb) / 2D
    val range = ub - lb
    Generation(List.fill(popSize) {
      val x = (range.toDenseVector :* DenseVector.rand(n, Uniform(-.5D,.5D))) + center
      Individual(f, x)
    })
  }

  private[optala] def selectParents(
      pop: Seq[Individual],
      strategy: SelectionStrategy,
      n: Int)(
      implicit randBasis: RandBasis = Rand): Seq[Individual] = strategy match {
    case FitnessProportionateSelection =>
      val minFitness = -1D*pop.map(_.objVal).max
      val dist = new Multinomial(Counter(pop.map(x => (x, -1D*x.objVal - minFitness + 1D))))
      dist.sample(n)
    case StochasticUniversalSampling =>
      val minFitness = pop.map(_.fitness).min
      // (individual, fitness normalized s.t. > 0)
      val popNorm = pop.map { case individual =>
        val normFitness =  individual.fitness - minFitness + 1D
        (individual, normFitness)
      }
      val f = popNorm.map(_._2).sum
      val stepSize = f / n

      val popNormSortedWithCumSums = popNorm
        .sortBy(-1D * _._2) // sort by descending fitness
        .foldRight(List[(Individual,Double)]()) {  // calculate cumulative sums of normalized fitness
          case ((individual, normFit), acc) =>
            acc match {
              case Nil => (individual, normFit) :: acc
              case (_,cumFitness)::_ => (individual, normFit + cumFitness) :: acc
            }
        }
        .reverse

      val start = new Uniform(0, stepSize).sample()
      (start to n*stepSize by stepSize).map { p =>
        val (individual, _) = popNormSortedWithCumSums.dropWhile(_._2 < p).head
        individual
      }
    case TournamentSelection(tournamentProp) =>
      require(
        0D <= tournamentProp && tournamentProp <= 1D,
        s"tournament proportion must be in [0,1] but got $tournamentProp")
      Seq.fill(n) {
        Random.shuffle(pop)
          .take(ceil(tournamentProp*pop.size).toInt)
          .maxBy(_.fitness)
      }
  }

  private[optala] def crossOver(f: Vector[Double] => Double, parents: Seq[Individual])(
      implicit randBasis: RandBasis = Rand): Seq[Individual] = {
    val dist = Uniform(0D, 1D)
    parents.grouped(2).map { x =>
      val p1 = x.head.point
      val p2 = x(1).point
      val t = dist.sample()
      val child = t*p1 + (1D - t)*p2
      Individual(f, child)
    }.toSeq
  }

  private[optala] def mutate(
      f: Vector[Double] => Double,
      lb: DenseVector[Double],
      ub: DenseVector[Double],
      parents: Seq[Solution],
      mutantCount: Int)(
      implicit randBasis: RandBasis = Rand): Seq[Individual] = {
    Random.shuffle(parents).take(mutantCount).map { x =>
      val delta = DenseVector.rand[Double](x.point.size, Gaussian(0, 1))
      val child = max(min(x.point + delta, ub), lb)
      Individual(f, child)
    }
  }
}

sealed trait SelectionStrategy
case object FitnessProportionateSelection extends SelectionStrategy
case object StochasticUniversalSampling extends SelectionStrategy
case class TournamentSelection(tournamentProportion: Double) extends SelectionStrategy
