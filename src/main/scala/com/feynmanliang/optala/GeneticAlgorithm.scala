package com.feynmanliang.optala

import scala.util.Random

import org.apache.commons.math3.random.MersenneTwister

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._

case class Generation(population: Seq[(Vector[Double],Double)]) {
  def meanNegFitness(): Double = population.map(_._2).sum / (1D*population.size)
  def bestIndividual(): (Vector[Double],Double) = population.minBy(_._2)
}

class GeneticAlgorithm(var maxSteps: Int = 1000) {
  /**
  *  Minimizes `f` subject to decision variables inside hypercube defined by `lb` and `ub`.
  *  TODO: perf diagnostics
  **/
  def minimize(
      f: Vector[Double] => Double,
      lb: Vector[Double],
      ub: Vector[Double],
      popSize: Int = 20,
      eliteCount: Int = 2,
      xoverFrac: Double = 0.8,
      seed: Option[Long] = None): (Option[Vector[Double]], Option[PerfDiagnostics[Generation]]) = {

    implicit val randBasis: RandBasis = seed match {
      case Some(s) => new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(s)))
      case None => new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister()))
    }
    val fCnt = new FunctionWithCounter(f)

    val xoverCount: Int = ((popSize - eliteCount) * xoverFrac).round.toInt
    val mutantCount: Int = popSize - eliteCount - xoverCount
    val initGen = initialize(f, lb, ub, popSize)

    val successors: Stream[Generation] = Stream.iterate(initGen) { gen =>
      val parents = selectParents(gen.population, min(2*xoverCount, mutantCount))

      val elites = gen.population.sortBy(_._2).take(eliteCount)
      val xovers = crossOver(f, parents)
      val mutants = mutate(f, lb, ub, parents, mutantCount)

      Generation(elites ++ xovers ++ mutants)
    }

    val iters = successors.take(maxSteps)
    val perf = PerfDiagnostics(
      iters.map(g => (g, g.meanNegFitness())),
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
    val range = (ub - lb)
    Generation(List.fill(popSize) {
      val x = (range :* (DenseVector.rand(n, Uniform(-.5D,.5D)))) + center
      (x, f(x))
    })
  }

  private[optala] def selectParents(
      pop: Seq[(Vector[Double],Double)], n: Int)(
      implicit randBasis: RandBasis = Rand): Seq[(Vector[Double],Double)] = {
    val minFitness = -1D*pop.map(_._2).max
    // TODO: fix multinomial random seed
    val dist = new Multinomial(Counter(pop.map(x => (x, -1D*x._2 - minFitness + 1D))))
    dist.sample(n)
  }

  private[optala] def crossOver(
      f: Vector[Double] => Double,
      parents: Seq[(Vector[Double],Double)])(
      implicit randBasis: RandBasis = Rand): Seq[(Vector[Double],Double)] = {
    val dist = Uniform(0D, 1D)
    parents.grouped(2).map { x =>
      val p1 = x(0)._1
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

// vim: set ts=2 sw=2 et sts=2:
