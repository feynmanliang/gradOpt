package com.feynmanliang.optala

import org.apache.commons.math3.random.MersenneTwister

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._

case class Generation(population: List[(Vector[Double], Double)]) {
  def meanFitness(): Double = population.map(_._2).sum / (1D*population.size)
}

class GeneticAlgorithm(var maxSteps: Int = 1000) {
  /** Minimizes `f` subject to decision variables inside hypercube defined by `lb` and `ub` */
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

    ???
  }

  private[optala] def initialize(
      f: Vector[Double] => Double, lb: Vector[Double], ub: Vector[Double], popSize: Int)(
      implicit randBasis: RandBasis): Generation = {
    val n = lb.size
    val center = (ub + lb) / 2D
    val range = (ub - lb)
    Generation(List.fill(popSize) {
      val x = (range :* (DenseVector.rand(n, Uniform(-.5D,.5D)(randBasis)))) + center
      (x, f(x))
    })
  }
}

// vim: set ts=2 sw=2 et sts=2:
