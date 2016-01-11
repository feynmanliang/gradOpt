package com.feynmanliang.optala.examples

import java.io.File

import breeze.linalg._
import breeze.numerics.pow
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator, Uniform}
import com.feynmanliang.optala.geneticalgorithm.{TournamentSelection, StochasticUniversalSampling, FitnessProportionateSelection, GeneticAlgorithm}
import com.feynmanliang.optala.neldermead.{NelderMeadOptimizer, Simplex}
import org.apache.commons.math3.random.MersenneTwister

import com.feynmanliang.optala._
import com.feynmanliang.optala.examples.ExampleUtils._

object GradientFreeExample {
  // This is the 6 Hump Camel Function (6HCF)
  val f: Vector[Double] => Double = v => {
    val x = v(0)
    val y = v(1)
    (4D - 2.1D * pow(x, 2) + (1D / 3D) * pow(x, 4)) * pow(x, 2) + x * y + (4D * pow(y, 2) - 4D) * pow(y, 2)
  }
  // Optimized over the region -2 <= x <= 2, -1 <= y <= 1
  val lb = DenseVector(-2D, -1D)
  val ub = DenseVector(2D, 1D)

  // Optimal points (http://www.sfu.ca/~ssurjano/camel6.html)
  val xOpts = List(
    DenseVector(-0.089842, 0.712656),
    DenseVector(0.089842, -0.712656)
  )
  val localMinima = xOpts ++ List(
    DenseVector(-1.70361, 0.796084),
    DenseVector(-1.6071, -0.568651),
    DenseVector(-1.23023-0.162335),
    DenseVector(1.23023, 0.162335),
    DenseVector(1.6071, 0.568651),
    DenseVector(1.70361, -0.796084)
  )
  val fOpt = -1.0316284534898774

  val seed = 42L
  implicit val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

  def main(args: Array[String]) {
    // Nelder-Mead Examples
    nmExample()
    nmObjEvalEff()
    nmConvRate()
    nmPerf()

    // Genetic Algorithm Examples
    gaExample()
    gaObjEvalEff()
    gaPerfPopSize()
    gaPerfEliteCount()
    gaPerfXoverFrac()
    gaPerfSelection()
  }
  def nmExample(): Unit = {
    println(s"===Optimizing 6HCF using Nedler-Mead===")
    val nmOpt = new NelderMeadOptimizer(maxObjEvals = 1000, maxIter = Int.MaxValue, tol = 0D)
    val initialSimplex: Simplex = createRandomSimplex(8, f)
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

  def nmObjEvalEff(): Unit = experimentWithResults("Nelder-Mead obj eval efficiency", "nm-obj-eval-eff.csv") {
    val n = 8
    val nmOpt = new NelderMeadOptimizer(maxObjEvals = 1000, maxIter = Int.MaxValue, tol = 0D)
    DenseMatrix.horzcat((for {
      _ <- 0 until 1000
    } yield {
      val initialSimplex = createRandomSimplex(n, f)
      val result = nmOpt.minimize(f, initialSimplex)
      val numIters = result.stateTrace.size.toDouble
      DenseMatrix(numIters)
    }): _*)
  }

  def nmConvRate(): Unit = {
    experimentWithResults("Nelder-Mead convergence rate, n=8", "nm-conv-rate.csv") {
      val n = 8
      val nmOpt = new NelderMeadOptimizer(maxObjEvals = Int.MaxValue, maxIter = Int.MaxValue, tol = 1E-6)
      DenseMatrix.horzcat((for {
        _ <- 0 until 1000
      } yield {
        val initialSimplex = createRandomSimplex(n, f)
        val result = nmOpt.minimize(f, initialSimplex)
        val numObjEval = result.numObjEval.toDouble
        DenseMatrix(numObjEval)
      }): _*)
    }

    experimentWithResults("Nelder-Mead convergence rate, varying n", "nm-conv-rate-vary-n.csv") {
      val nmOpt = new NelderMeadOptimizer(maxObjEvals = Int.MaxValue, maxIter = Int.MaxValue, tol = 1E-6)
      DenseMatrix.horzcat((for {
        n <- 3 to 30
        _ <- 0 until 1000
      } yield {
        val initialSimplex = createRandomSimplex(n, f)
        val result = nmOpt.minimize(f, initialSimplex)
        val numObjEval = result.numObjEval.toDouble
        DenseMatrix(n.toDouble, numObjEval)
      }): _*)
    }
  }

  def nmPerf(): Unit = experimentWithResults(s"Nelder-Mead number simplex points", "nm-vary-n.csv") {
    val nmOpt = new NelderMeadOptimizer(maxObjEvals = 1000, maxIter = Int.MaxValue, tol = 0D)
    DenseMatrix.horzcat((for {
      n <- 3 to 30
      _ <- 0 until 1000
    } yield {
      val initialSimplex = createRandomSimplex(n, f)
      val result = nmOpt.minimize(f, initialSimplex)
      val xStar = result.bestSolution
      val fStar = xStar.objVal
      val bias = fStar - fOpt
      val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar.point)))

      DenseMatrix(n.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
    }): _*)
  }

  def gaExample(): Unit = {
    println(s"===Optimizing 6HCF using GA===")
    val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxIter = Int.MaxValue)
    val popSize = 20
    val eliteCount = 2
    val xoverFrac = 0.8
    val result = ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, None)
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

  def gaObjEvalEff(): Unit = experimentWithResults("GA obj eval efficiency", "ga-obj-eval-eff.csv"){
    val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxIter = Int.MaxValue)
    val popSize = 20
    val eliteCount = 2
    val xoverFrac = 0.8
    DenseMatrix.horzcat((for {
      _ <- 0 until 1000
    } yield {
      val result = ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, None)
      val numGens = result.stateTrace.size
      DenseMatrix(numGens.toDouble)
    }): _*)
  }

  def gaPerfPopSize(): Unit = {
    experimentWithResults("GA population size", "ga-pop-size.csv") {
      val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxIter = Int.MaxValue)
      val popSize = 20
      val eliteCount = 2
      val xoverFrac = 0.8
      DenseMatrix.horzcat((for {
        popSize <- 3 to 50
        _ <- 0 until 1000
      } yield {
        val result = ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, None)
        val best = result.bestSolution
        val (xStar, fStar) = (best.point, best.objVal)
        val bias = fStar - fOpt
        val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))

        DenseMatrix(popSize.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
      }): _*)
    }

    experimentWithResults("GA population size - num iters", "ga-pop-size-num-iters.csv") {
      val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxIter = Int.MaxValue)
      val popSize = 20
      val eliteCount = 2
      val xoverFrac = 0.8
      DenseMatrix.horzcat((for {
        popSize <- 3 to 50
      } yield {
        val result = ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, None)
        val numIters = result.stateTrace.size.toDouble
        DenseMatrix(popSize.toDouble, numIters)
      }): _*)
    }

    // Save final states to show budget exhausted before convergence
    // columns = (popSize, x1,y1,x2,y2,...), rows = observations
    experimentWithResults("GA population size - states", "ga-pop-size-final-states.csv") {
      val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxIter = Int.MaxValue)
      val popSize = 20
      val eliteCount = 2
      val xoverFrac = 0.8
      DenseMatrix.horzcat((for {
        popSize <- List(3, 7, 20, 50)
      } yield {
        val result = ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, None)
        val numIters = result.stateTrace.size.toDouble
        DenseMatrix(popSize.toDouble +: result.stateTrace.last.population.flatMap(_.point.toArray).toArray)
      }): _*)
    }
  }

  def gaPerfEliteCount(): Unit = {
    experimentWithResults("GA elite count", "ga-elite-count.csv") {
      val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxIter = Int.MaxValue)
      val popSize = 20
      val eliteCount = 2
      val xoverFrac = 0.8
      DenseMatrix.horzcat((for {
        eliteCount <- 0 until 20
        _ <- 0 until 1000
      } yield {
        val result = ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, None)
        val best = result.bestSolution
        val (xStar, fStar) = (best.point, best.objVal)
        val bias = fStar - fOpt
        val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))

        DenseMatrix(eliteCount.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
      }): _*)
    }

    experimentWithResults("GA elite count - 0 elite count => no monotonicity", "ga-elite-count-0.csv") {
      val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxIter = Int.MaxValue)
      val popSize = 20
      val eliteCount = 0
      val xoverFrac = 0.8
      val result = ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, None)
      //min population objective value
      DenseMatrix(result.stateTrace.map(_.bestIndividual.objVal): _*)
    }

    experimentWithResults("GA elite count - num iters", "ga-elite-count-num-iters.csv") {
      val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxIter = Int.MaxValue)
      val popSize = 20
      val eliteCount = 2
      val xoverFrac = 0.8
      DenseMatrix.horzcat((for {
        eliteCount <- 0 until 20
      } yield {
        val result = ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, None)
        val numIters = result.stateTrace.size.toDouble
        DenseMatrix(eliteCount.toDouble, numIters)
      }): _*)
    }
  }

  def gaPerfXoverFrac(): Unit = {
    experimentWithResults("GA xover frac", "ga-xover-frac.csv") {
        val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxIter = Int.MaxValue)
        val popSize = 20
        val eliteCount = 2
        val xoverFrac = 0.8
        DenseMatrix.horzcat((for {
          xoverFrac <- 0.0 to 1.0 by 0.05
          _ <- 0 until 1000
        } yield {
          val result = ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, None)
          val best = result.bestSolution
          val (xStar, fStar) = (best.point, best.objVal)
          val bias = fStar - fOpt
          val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))

          DenseMatrix(xoverFrac.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
        }): _*)
      }
    experimentWithResults("GA xover frac - noElite", "ga-xover-frac-noElite.csv") {
        val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxIter = Int.MaxValue)
        val popSize = 20
        val eliteCount = 0
        val xoverFrac = 0.8
        DenseMatrix.horzcat((for {
          xoverFrac <- 0.0 to 1.0 by 0.05
          _ <- 0 until 1000
        } yield {
          val result = ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, None)
          val best = result.bestSolution
          val (xStar, fStar) = (best.point, best.objVal)
          val bias = fStar - fOpt
          val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))

          DenseMatrix(xoverFrac.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
        }): _*)
      }

  }
  def gaPerfSelection(): Unit = {
    experimentWithResults("GA selection schemes: non-tournament", "ga-selection.csv") {
      val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxIter = Int.MaxValue)
      val popSize = 20
      val eliteCount = 2
      val xoverFrac = 0.8

      val schemes = List(FitnessProportionateSelection, StochasticUniversalSampling)
      DenseMatrix.horzcat((for {
        i <- schemes.indices
        _ <- 0 until 1000
      } yield {
        val scheme = schemes(i)

        val result = ga.minimize(f, lb, ub, popSize, scheme, eliteCount, xoverFrac, None)
        val best = result.bestSolution
        val (xStar, fStar) = (best.point, best.objVal)
        val bias = fStar - fOpt
        val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))

        DenseMatrix(i.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
      }): _*)
    }

    experimentWithResults("GA selection schemes: tournament", "ga-selection-tournament.csv") {
      val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxIter = Int.MaxValue)
      val popSize = 20
      val eliteCount = 2
      val xoverFrac = 0.8

      DenseMatrix.horzcat((for {
        tournamentProb <- 0.05D until 0.95D by 0.05D
        _ <- 0 until 1000
      } yield {
        val result = ga.minimize(f, lb, ub, popSize, TournamentSelection(tournamentProb), eliteCount, xoverFrac, None)
        val best = result.bestSolution
        val (xStar, fStar) = (best.point, best.objVal)
        val bias = fStar - fOpt
        val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))

        DenseMatrix(tournamentProb.toDouble, fStar, bias, if (closestToGlobal) 1D else 0D)
      }): _*)
    }
  }
}
