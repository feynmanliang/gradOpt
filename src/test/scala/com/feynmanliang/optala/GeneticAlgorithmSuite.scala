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

      it("initializes the correct population size") {
        assert(init.population.size === popSize)
      }
      it("respects the bounds") {
        assert(init.population.map(_._1).forall(x => all((lb :<= x) :& (x :<= ub))))
      }
    }

    describe("selectParents") {
      val init = ga.initialize(f, lb, ub, popSize)

      it("selects the number of specified parents") {
        assert(ga.selectParents(init.population, 10).size === 10)
      }
      it("samples with replacement") {
        val parents = ga.selectParents(init.population, popSize*2)
        assert(parents.toSet.size <= (parents.size + 1) / 2)
      }
    }

    describe("crossover") {
      val init = ga.initialize(f, lb, ub, popSize)

      it("halves the number of parents") {
        assert(ga.crossOver(f, init.population).size === init.population.size / 2)
      }
      it("creates distinct children") {
        assert((ga.crossOver(f, init.population) ++ init.population).toSet.size >= (init.population.size * 1.5D - 1).toInt)
      }
    }

    describe("mutate") {
      val init = ga.initialize(f, lb, ub, popSize)

      val mutants = ga.mutate(f, lb, ub, init.population, 5)
      it("only keeps mutantCount elements") {
        assert(mutants.size == 5)
      }
      it("creates distinct children") {
        assert(mutants.toSet.size === mutants.size)
      }
      it("stays within bounds") {
        assert(mutants.map(_._1).forall(x => all((lb :<= x) :& (x :<= ub))))
      }
    }

    describe("when run on six-hump camelback function (6HCF)") {
      ga.minimize(f, lb, ub, popSize=popSize, eliteCount=2, xoverFrac=0.8, seed=Some(seed)) match {
        case (_, Some(perf)) =>
          it("monotonically decreases the best point's objective") {
            assert(perf.stateTrace.map(_.bestIndividual()._2).sliding(2).forall(x => x.head >= x(1)))
          }
          it("decreases average objective value over all population") {
            val avgObjTrace = perf.stateTrace.map(_.meanNegFitness())
            assert(avgObjTrace.take(10).sum >= avgObjTrace.takeRight(10).sum)
          }
        case _ => fail("error")
      }
    }
  }
}
