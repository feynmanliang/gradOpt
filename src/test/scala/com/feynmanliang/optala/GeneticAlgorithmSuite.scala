package com.feynmanliang.optala

import org.apache.commons.math3.random.MersenneTwister
import org.scalatest._

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._


class GeneticAlgorithmSuite extends FunSpec {
  val seed = 42
  implicit val randBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

  describe("GeneticAlgorithm") {
    val f: Vector[Double] => Double = v => {
      val x = v(0)
      val y = v(1)
      (4D - 2.1D*pow(x,2) + (1D/3D)*pow(x,4))*pow(x,2) + x*y + (4D*pow(y,2) - 4D)*pow(y,2)
    }
    val lb = DenseVector(-2D, -1D)
    val ub = DenseVector(2D, 1D)
    val popSize = 20
    val ga = new GeneticAlgorithm(maxSteps = 1000)

    describe("initialize") {
      val init = ga.initialize(f, lb, ub, popSize)

      println(init)
      it("initializes the correct population size") {
        assert(init.population.size === popSize)
      }
      it("respects the bounds") {
        assert(init.population.map(_._1).forall(x => ((lb :<= x) :& (x :<= ub)).all))
      }
    }

    describe("when run on six-hump camelback function (6HCF)") {
      it("runs") {
        println(ga.minimize(f, lb, ub, popSize=popSize, eliteCount=2, xoverFrac=0.8, seed=Some(seed)))
      }
    }
  }
}

// vim: set ts=2 sw=2 et sts=2:
