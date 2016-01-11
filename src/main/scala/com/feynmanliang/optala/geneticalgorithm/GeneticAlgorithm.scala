package com.feynmanliang.optala.geneticalgorithm

import scala.util.Random

import breeze.linalg._
import breeze.stats.distributions._

import com.feynmanliang.optala.{FunctionWithCounter, RunResult, Solution}

/** An individual (i.e. a candidate `Solution`). */
private[optala] case class Individual(
    override val f: Vector[Double] => Double,
    override val point: DenseVector[Double]) extends Solution(f, point) {
  val fitness = -1D * objVal // a GA maximizes fitness := -1*objVal <=> minimizes objVal
}

/** A generation (i.e. iteration of a GA).
  * @param population the individuals alive in this generation
  */
private[optala] case class Generation(population: Seq[Individual]) {
  def averageObjVal: Double = population.map(_.objVal).sum / population.size
  def averageFitness: Double = -1D * averageObjVal
  val bestIndividual = population.minBy(_.objVal)
}

/** Results from a run of a GA. */
private[optala] case class GARunResult(
    override val stateTrace: List[Generation],
    override val numObjEval: Long,
    override val numGradEval: Long) extends RunResult[Generation] {
  override val bestSolution = stateTrace.maxBy(_.bestIndividual.fitness).bestIndividual
}

/** An implementation of the Genetic Algorithm (GA) optimization method for constrained optimizations over a hypercube
  * feasible set.
  * @param maxObjectiveEvals maximum number of objective function evaluations before termination
  * @param maxIter maximum number of iterations before termination
  */
class GeneticAlgorithm(
    var maxObjectiveEvals: Int = Int.MaxValue,
    var maxIter: Int = 1000) {

  /** Minimizes `f` subject to decision variables inside hypercube defined by `lb` and `ub`.
    * @param f objective function
    * @param lb lower bounds of hypercube
    * @param ub upper bounds of hypercube
    * @param popSize population size
    * @param selectionStrategy selection strategy
    * @param eliteCount elite count
    * @param xoverFrac crossover fraction
    * @param randBasis seed for Breeze random number generator
    */
  def minimize(
      f: Vector[Double] => Double,
      lb: DenseVector[Double],
      ub: DenseVector[Double],
      popSize: Int = 20,
      selectionStrategy: SelectionStrategy = FitnessProportionateSelection,
      eliteCount: Int = 2,
      xoverFrac: Double = 0.8)(implicit randBasis: RandBasis = Rand): RunResult[Generation] = {
    val fCnt = new FunctionWithCounter(f)

    val xoverCount: Int = ((popSize - eliteCount) * xoverFrac).ceil.toInt
    val mutantCount: Int = popSize - eliteCount - xoverCount
    val initGen = initialize(fCnt, lb, ub, popSize)

    val successors: Stream[Generation] = Stream.iterate(initGen) { gen =>
      val numParents = ((max(2*xoverCount, mutantCount)+1) / 2) * 2 // ensure even number parents
      val parents = selectionStrategy.selectParents(gen.population, numParents)

      val elites = gen.population.sortBy(_.objVal).take(eliteCount)
      val parentPairs = parents.grouped(2).map(x => (x.head, x(1))).toSeq
      val xovers = crossOver(fCnt, parentPairs)
      val mutants = mutate(fCnt, lb, ub, parents, mutantCount)

      Generation(elites ++ xovers ++ mutants)
    }

    val trace = successors
      .takeWhile(_ => fCnt.numCalls <= maxObjectiveEvals)
      .take(maxIter)
      .toList
    GARunResult(trace, fCnt.numCalls, 0)
  }

  /** Initializes a population to randomly lie within the hypercube given by `lb` and `ub`. */
  private[optala] def initialize(
      f: Vector[Double] => Double, lb: DenseVector[Double], ub: DenseVector[Double], popSize: Int)(
      implicit randBasis: RandBasis): Generation = {
    val n = lb.size
    val center = (ub + lb) / 2D
    val range = ub - lb
    Generation(List.fill(popSize) {
      val x = (range.toDenseVector :* DenseVector.rand(n, Uniform(-.5D,.5D))) + center
      Individual(f, x)
    })
  }

  /** Crosses over pairs of parents by taking a random convex combination.
    * @param f objective function
    * @param parentPairs parent pairs to cross over
    */
  private[optala] def crossOver(f: Vector[Double] => Double, parentPairs: Seq[(Individual, Individual)])(
      implicit randBasis: RandBasis): Seq[Individual] = {
    parentPairs.map { p =>
      val t = Uniform(0D, 1D).sample()
      val child = t*p._1.point + (1D - t)*p._2.point
      Individual(f, child)
    }
  }

  /** Generates mutant individuals by mutating parents. */
  private[optala] def mutate(
      f: Vector[Double] => Double,
      lb: DenseVector[Double],
      ub: DenseVector[Double],
      parents: Seq[Solution],
      mutantCount: Int)(
      implicit randBasis: RandBasis): Seq[Individual] = {
    Random.setSeed(randBasis.randInt.sample())
    Random.shuffle(parents).take(mutantCount).map { x =>
      val delta = DenseVector.rand[Double](x.point.size, Gaussian(0, 1))
      val child = max(min(x.point + delta, ub), lb)
      Individual(f, child)
    }
  }
}
