package com.feynmanliang.optala.examples

import java.io.File

import breeze.linalg._
import breeze.numerics.pow
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import com.feynmanliang.optala.examples.ExampleUtils._
import com.feynmanliang.optala.geneticalgorithm.{FitnessProportionateSelection, GeneticAlgorithm, StochasticUniversalSampling, TournamentSelection}
import com.feynmanliang.optala.neldermead.{NelderMead, Simplex}
import org.apache.commons.math3.random.MersenneTwister

object GradientFreeExample {
  // 6 Hump Camel Function (6HCF)
  val f: Vector[Double] => Double = v => {
    val x = v(0)
    val y = v(1)
    (4D - 2.1D * pow(x, 2) + (1D / 3D) * pow(x, 4)) * pow(x, 2) + x * y + (4D * pow(y, 2) - 4D) * pow(y, 2)
  }

  // Feasible region: -2 <= x <= 2, -1 <= y <= 1
  val lb = DenseVector(-2D, -1D)
  val ub = DenseVector(2D, 1D)

  // Global minima
  val xOpts = List(
    DenseVector(-0.089842, 0.712656),
    DenseVector(0.089842, -0.712656)
  )

  // Local minima
  val localMinima = xOpts ++ List(
    DenseVector(-1.70361, 0.796084),
    DenseVector(-1.6071, -0.568651),
    DenseVector(-1.23023, -0.162335),
    DenseVector(1.23023, 0.162335),
    DenseVector(1.6071, 0.568651),
    DenseVector(1.70361, -0.796084)
  )

  // Minimum objective value
  val fOpt = -1.0316284534898774

  val SEED = 43L // random seed
  val NUM_TRIALS = 3
  val MAX_OBJ_EVALS = 100

  def main(args: Array[String]) {
    // Nelder-Mead
    nmExample()
    nmObjEvalEff()
    nmConvRate()
    nmPerf()

    // Genetic Algorithm
    gaExample()
    gaObjEvalEff()
    gaPerfPopSize()
    gaPerfEliteCount()
    gaPerfXoverFrac()
    gaPerfSelection()
  }

  def nmExample(): Unit = {
    val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(SEED)))

    println(s"===Optimizing 6HCF using Nedler-Mead===")
    val nmOpt = new NelderMead(maxObjEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue, tol = 0D)
    val initialSimplex: Simplex = createRandomSimplex(n = 8, f)(rand)
    val result = nmOpt.minimize(f, initialSimplex)
    val xStar = result.bestSolution

    // columns = (x1,y1,x2,y2,...), rows = iterations
    val stateTrace = DenseMatrix.vertcat(result.stateTrace.map { simplex =>
      DenseMatrix(simplex.solutionsByObj.flatMap(_.point.toArray))
    }: _*)
    val stateTraceFile = new File("results/nm-stateTrace.csv")
    csvwrite(stateTraceFile, stateTrace)
    println(s"Wrote stateTrace to $stateTraceFile")

    // objective value at simplex centroid
    val objTrace = DenseMatrix(result.stateTrace.map { s =>
      val candidates = s.solutionsByObj.map(_.point)
      f(candidates.reduce(_+_) / candidates.size.toDouble)
    }: _*)
    val objTraceFile = new File("results/nm-objTrace.csv")
    csvwrite(objTraceFile, objTrace)
    println(s"Wrote objTrace to $objTraceFile")
  }

  def nmObjEvalEff(): Unit = experimentWithResults("Nelder-Mead obj eval efficiency, n=8", SEED, "nm-obj-eval-eff.csv") { seedRB =>
    val seedRB = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(SEED)))
    val nmOpt = new NelderMead(maxObjEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue, tol = 0D)
    DenseMatrix.horzcat((for {
      _ <- 0 until NUM_TRIALS
      seed = seedRB.randInt.sample()
    } yield {
      val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
      val initialSimplex = createRandomSimplex(n = 8, f)(rand)
      val result = nmOpt.minimize(f, initialSimplex)
      val numIters = result.stateTrace.size.toDouble
      DenseMatrix(numIters)
    }): _*)
  }

  def nmConvRate(): Unit = {
    experimentWithResults("Nelder-Mead convergence rate, n=8", SEED, "nm-conv-rate.csv") { seedRB =>
      val nmOpt = new NelderMead(maxObjEvals = Int.MaxValue, maxIter = Int.MaxValue, tol = 1E-6)
      DenseMatrix.horzcat((for {
        _ <- 0 until NUM_TRIALS
        seed = seedRB.randInt.sample()
      } yield {
        val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
        val initialSimplex = createRandomSimplex(n = 8, f)(rand)
        val result = nmOpt.minimize(f, initialSimplex)
        val numObjEval = result.numObjEval.toDouble
        DenseMatrix(numObjEval)
      }): _*)
    }
    experimentWithResults("Nelder-Mead convergence rate, varying n", SEED, "nm-conv-rate-vary-n.csv") { seedRB =>
      val seedRB = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(SEED)))
      val nmOpt = new NelderMead(maxObjEvals = Int.MaxValue, maxIter = Int.MaxValue, tol = 1E-6)
      DenseMatrix.horzcat((for {
        _ <- 0 until NUM_TRIALS
        seed = seedRB.randInt.sample()
        n <- 3 to 30
      } yield {
        val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
        val initialSimplex = createRandomSimplex(n, f)(rand)
        val result = nmOpt.minimize(f, initialSimplex)
        val numObjEval = result.numObjEval.toDouble
        DenseMatrix(n.toDouble, numObjEval)
      }): _*)
    }
  }

  def nmPerf(): Unit = experimentWithResults(s"Nelder-Mead number simplex points", SEED, "nm-vary-n.csv") { seedRB =>
    val seedRB = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(SEED)))
    val nmOpt = new NelderMead(maxObjEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue, tol = 0D)
    DenseMatrix.horzcat((for {
      _ <- 0 until NUM_TRIALS
      seed = seedRB.randInt.sample()
      n <- 3 to 30
    } yield {
      val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
      val initialSimplex = createRandomSimplex(n, f)(rand)
      val result = nmOpt.minimize(f, initialSimplex)
      val xStar = result.bestSolution
      val fStar = xStar.objVal
      val bias = fStar - fOpt
      val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar.point)))
      DenseMatrix(n.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
    }): _*)
  }

  def gaExample(): Unit = {
    val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(SEED)))

    println(s"===Optimizing 6HCF using GA===")
    val ga = new GeneticAlgorithm(maxObjectiveEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue)
    val result = ga.minimize(
      f, lb, ub,
      popSize = 20,
      StochasticUniversalSampling,
      eliteCount = 2,
      xoverFrac = 0.8)(rand)
    val best = result.bestSolution
    val (xstar, fstar) = (best.point, best.objVal)

    // columns = (x1,y1,x2,y2,...), rows = iterations
    val stateTrace = DenseMatrix.vertcat(result.stateTrace.map { gen =>
      DenseMatrix(gen.population.flatMap(_.point.toArray))
    }: _*)
    val stateTraceFile = new File("results/ga-stateTrace.csv")
    csvwrite(stateTraceFile, stateTrace)
    println(s"Wrote stateTrace to $stateTraceFile")

    // mean population objective value
    val meanObjTrace = DenseMatrix(result.stateTrace.map(_.averageObjVal): _*)
    // min population objective value
    val minObjTrace = DenseMatrix(result.stateTrace.map(_.bestIndividual.objVal): _*)
    val objTraceFile = new File("results/ga-objTrace.csv")
    csvwrite(objTraceFile, DenseMatrix.horzcat(meanObjTrace, minObjTrace))
    println(s"Wrote objTraces to $objTraceFile")
  }

  def gaObjEvalEff(): Unit = experimentWithResults("GA obj eval efficiency", SEED, "ga-obj-eval-eff.csv") { seedRB =>
    val ga = new GeneticAlgorithm(maxObjectiveEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue)
    DenseMatrix.horzcat((for {
      _ <- 0 until NUM_TRIALS
      seed = seedRB.randInt.sample()
    } yield {
      val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
      val result = ga.minimize(
        f, lb, ub,
        popSize = 20,
        StochasticUniversalSampling,
        eliteCount = 2,
        xoverFrac = 0.8)(rand)
      val numGens = result.stateTrace.size
      DenseMatrix(numGens.toDouble)
    }): _*)
  }

  def gaPerfPopSize(): Unit = {
    experimentWithResults("GA population size", SEED, "ga-pop-size.csv") { seedRB =>
      val ga = new GeneticAlgorithm(maxObjectiveEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue)
      DenseMatrix.horzcat((for {
        _ <- 0 until NUM_TRIALS
        seed = seedRB.randInt.sample()
        popSize <- 3 to 50
      } yield {
        val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
        val result = ga.minimize(
          f, lb, ub,
          popSize = popSize,
          StochasticUniversalSampling,
          eliteCount = 2,
          xoverFrac = 0.8)(rand)
        val best = result.bestSolution
        val (xStar, fStar) = (best.point, best.objVal)
        val bias = fStar - fOpt
        val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))
        DenseMatrix(popSize.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
      }): _*)
    }
    experimentWithResults("GA population size - num iters", SEED, "ga-pop-size-num-iters.csv") { seedRB =>
      val ga = new GeneticAlgorithm(maxObjectiveEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue)
      val seed = seedRB.randInt.sample()
      DenseMatrix.horzcat((for {
        popSize <- 3 to 50
      } yield {
        val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
        val result = ga.minimize(
          f, lb, ub,
          popSize = popSize,
          StochasticUniversalSampling,
          eliteCount = 2,
          xoverFrac = 0.8)(rand)
        val numIters = result.stateTrace.size.toDouble
        DenseMatrix(popSize.toDouble, numIters)
      }): _*)
    }

    // Save final states to show budget exhausted before convergence
    // columns = (popSize, x1,y1,x2,y2,...), rows = observations
    experimentWithResults("GA population size - states", SEED, "ga-pop-size-final-states.csv") { seedRB =>
      val ga = new GeneticAlgorithm(maxObjectiveEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue)
      val seed = seedRB.randInt.sample()
      DenseMatrix.horzcat((for {
        popSize <- List(3, 7, 20, 50)
      } yield {
        val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
        val result = ga.minimize(
          f, lb, ub,
          popSize = popSize,
          StochasticUniversalSampling,
          eliteCount = 2,
          xoverFrac = 0.8)(rand)
        val numIters = result.stateTrace.size.toDouble
        DenseMatrix(popSize.toDouble +: result.stateTrace.last.population.flatMap(_.point.toArray).toArray)
      }): _*)
    }
  }

  def gaPerfEliteCount(): Unit = {
    experimentWithResults("GA elite count", SEED, "ga-elite-count.csv") { seedRB =>
      val ga = new GeneticAlgorithm(maxObjectiveEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue)
      DenseMatrix.horzcat((for {
        _ <- 0 until NUM_TRIALS
        seed = seedRB.randInt.sample()
        eliteCount <- 0 until 20
      } yield {
        val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
        val result = ga.minimize(
          f, lb, ub,
          popSize = 20,
          StochasticUniversalSampling,
          eliteCount = eliteCount,
          xoverFrac = 0.8)(rand)
        val best = result.bestSolution
        val (xStar, fStar) = (best.point, best.objVal)
        val bias = fStar - fOpt
        val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))
        DenseMatrix(eliteCount.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
      }): _*)
    }
    experimentWithResults("GA elite count - 0 elite count => no monotonicity", SEED, "ga-elite-count-0.csv") { seedRB =>
      val ga = new GeneticAlgorithm(maxObjectiveEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue)
      val seed = seedRB.randInt.sample()
      val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
      val result = ga.minimize(
        f, lb, ub,
        popSize = 20,
        StochasticUniversalSampling,
        eliteCount = 0,
        xoverFrac = 0.8)(rand)
      //min population objective value
      DenseMatrix(result.stateTrace.map(_.bestIndividual.objVal): _*)
    }
    experimentWithResults("GA elite count - num iters", SEED, "ga-elite-count-num-iters.csv") { seedRB =>
      val ga = new GeneticAlgorithm(maxObjectiveEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue)
      val seed = seedRB.randInt.sample()
      val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
      DenseMatrix.horzcat((for {
        eliteCount <- 0 until 20
      } yield {
        val result = ga.minimize(
          f, lb, ub,
          popSize = 20,
          StochasticUniversalSampling,
          eliteCount = eliteCount,
          xoverFrac = 0.8)(rand)
        val numIters = result.stateTrace.size.toDouble
        DenseMatrix(eliteCount.toDouble, numIters)
      }): _*)
    }
  }

  def gaPerfXoverFrac(): Unit = {
    experimentWithResults("GA xover frac", SEED, "ga-xover-frac.csv") { seedRB =>
      val ga = new GeneticAlgorithm(maxObjectiveEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue)
      DenseMatrix.horzcat((for {
        _ <- 0 until NUM_TRIALS
        seed = seedRB.randInt.sample()
        xoverFrac <- 0.0 to 1.0 by 0.05
      } yield {
        val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
        val result = ga.minimize(
          f, lb, ub,
          popSize = 20,
          selectionStrategy = StochasticUniversalSampling,
          eliteCount = 2,
          xoverFrac = xoverFrac)(rand)
        val best = result.bestSolution
        val (xStar, fStar) = (best.point, best.objVal)
        val bias = fStar - fOpt
        val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))
        DenseMatrix(xoverFrac.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
      }): _*)
    }
    experimentWithResults("GA xover frac - noElite", SEED, "ga-xover-frac-noElite.csv") { seedRB =>
      val ga = new GeneticAlgorithm(maxObjectiveEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue)
      DenseMatrix.horzcat((for {
        _ <- 0 until NUM_TRIALS
        seed = seedRB.randInt.sample()
        xoverFrac <- 0.0 to 1.0 by 0.05
      } yield {
        val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
        val result = ga.minimize(
          f, lb, ub,
          popSize = 20,
          selectionStrategy = StochasticUniversalSampling,
          eliteCount = 0,
          xoverFrac = 0.8)(rand)
        val best = result.bestSolution
        val (xStar, fStar) = (best.point, best.objVal)
        val bias = fStar - fOpt
        val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))
        DenseMatrix(xoverFrac.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
      }): _*)
    }
  }

  def gaPerfSelection(): Unit = {
    experimentWithResults("GA selection schemes: non-tournament", SEED, "ga-selection.csv") { seedRB =>
      val ga = new GeneticAlgorithm(maxObjectiveEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue)
      val strategies = List(FitnessProportionateSelection, StochasticUniversalSampling)
      DenseMatrix.horzcat((for {
        _ <- 0 until NUM_TRIALS
        seed = seedRB.randInt.sample()
        i <- strategies.indices
      } yield {
        val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
        val result = ga.minimize(
          f, lb, ub,
          popSize = 20,
          selectionStrategy = strategies(i),
          eliteCount = 2,
          xoverFrac = 0.8)(rand)
        val best = result.bestSolution
        val (xStar, fStar) = (best.point, best.objVal)
        val bias = fStar - fOpt
        val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))
        DenseMatrix(i.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
      }): _*)
    }
    experimentWithResults("GA selection schemes: tournament", SEED, "ga-selection-tournament.csv") { seedRB =>
      val ga = new GeneticAlgorithm(maxObjectiveEvals = MAX_OBJ_EVALS, maxIter = Int.MaxValue)
      DenseMatrix.horzcat((for {
        _ <- 0 until NUM_TRIALS
        seed = seedRB.randInt.sample()
        tournamentProb <- 0.05D until 0.95D by 0.05D
      } yield {
        val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
        val result = ga.minimize(
          f, lb, ub,
          popSize = 20,
          selectionStrategy = TournamentSelection(tournamentProb),
          eliteCount = 2,
          xoverFrac = 0.8)(rand)
        val best = result.bestSolution
        val (xStar, fStar) = (best.point, best.objVal)
        val bias = fStar - fOpt
        val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))
        DenseMatrix(tournamentProb.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
      }): _*)
    }
  }
}
