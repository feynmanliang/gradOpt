package com.feynmanliang.optala.geneticalgorithm

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import org.scalatest._

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
    val maxObjEvals = 1000
    val ga = new GeneticAlgorithm(maxObjectiveEvals = maxObjEvals)

    describe("initialize") {
      val init = ga.initialize(f, lb, ub, popSize)

      it("initializes the correct population size") {
        assert(init.population.size === popSize)
      }
      it("respects the bounds") {
        assert(init.population.map(_.point).forall(x => all((lb :<= x) :& (x :<= ub))))
      }
    }

    describe("crossover") {
      val init = ga.initialize(f, lb, ub, popSize)
      val parentPairs = init.population.grouped(2).map(x => (x.head, x(1))).toSeq

      it("halves the number of parents") {
        assert(ga.crossOver(f, parentPairs).size === init.population.size / 2)
      }
      it("creates distinct children") {
        assert((ga.crossOver(f, parentPairs) ++ init.population).toSet.size >= (init.population.size * 1.5D - 1).toInt)
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
        assert(mutants.map(_.point).forall(x => all((lb :<= x) :& (x :<= ub))))
      }
    }

    describe("when run on six-hump camelback function (6HCF)") {
      val result = ga.minimize(f, lb, ub, popSize=popSize, eliteCount=2, xoverFrac=0.8)
      it("monotonically decreases the best point's objective") {
        assert(result.stateTrace.map(_.bestIndividual.objVal).sliding(2).forall(x => x.head >= x(1)))
      }
      it("decreases average objective value over all population") {
        val avgObjTrace = result.stateTrace.map(_.averageObjVal)
        assert(avgObjTrace.take(10).sum >= avgObjTrace.takeRight(10).sum)
      }
      it(s"terminates after evaluating objective function $maxObjEvals times") {
        // assumes each iteration evaluates objective less that 0.2*maxObjectiveEvals
        assert(result.numObjEval <= (maxObjEvals*1.2).toInt)
      }
    }
  }
  describe("SelectionStrategy") {
    val seed = 42
    implicit val randBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

    val f: Vector[Double] => Double = v => {
      val x = v(0)
      val y = v(1)
      (4D - 2.1D*pow(x,2) + (1D/3D)*pow(x,4))*pow(x,2) + x*y + (4D*pow(y,2) - 4D)*pow(y,2)
    }
    val lb = DenseVector(-2D, -1D)
    val ub = DenseVector(2D, 1D)
    val popSize = 20
    val maxObjEvals = 1000
    val ga = new GeneticAlgorithm(maxObjectiveEvals = maxObjEvals)

    describe("selectParents") {
      val init = ga.initialize(f, lb, ub, popSize)

      for {
        strategy <- List(FitnessProportionateSelection, StochasticUniversalSampling, TournamentSelection(0.5))
      } {
        describe(s"$strategy") {
          it("selects the number of specified parents") {
            assert(strategy.selectParents(init.population, 10).size === 10)
          }
          it("samples with replacement") {
            val parents = strategy.selectParents(init.population, popSize*2)
            assert(parents.toSet.size <= (parents.size + 1) / 2)
          }
          it("selects individuals with higher than average fitness") {
            val selected = strategy.selectParents(init.population, popSize*2)
            val selectedAvgFitness = selected.map(_.fitness).sum / selected.size.toDouble
            assert(selectedAvgFitness >= init.averageFitness)
          }
        }
      }
    }
  }
}
