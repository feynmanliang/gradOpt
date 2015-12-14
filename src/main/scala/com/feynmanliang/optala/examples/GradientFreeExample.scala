package com.feynmanliang.optala.examples

import java.io.File

import org.apache.commons.math3.random.MersenneTwister

import breeze.linalg._
import breeze.numerics.pow
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator, Uniform}

import com.feynmanliang.optala._

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
//    nmExampleN8()
//    nmObjEvalEff()
//    nmConvRate()
//    nmPerf()

    // Genetic Algorithm Examples
//    gaExample()
//    gaObjEvalEff()
//    gaPerfPopSize()
//    gaPerfEliteCount()
//    gaPerfXoverFrac()
    gaPerfSelection()
  }

  def nmObjEvalEff(): Unit = experimentWithResults("Nelder-Mead obj eval efficiency", "nm-obj-eval-eff.csv") {
    val n = 8
    val nmOpt = new NelderMeadOptimizer(maxObjectiveEvals = 1000, maxSteps = Int.MaxValue, tol = 0D)
    DenseMatrix.horzcat((for {
      _ <- 0 until 1000
    } yield {
      val initialSimplex = Simplex(Seq.fill(n) {
        val simplexPoint = DenseVector(Uniform(-2D, 2D).sample(), Uniform(-1D, 1D).sample())
        (simplexPoint, f(simplexPoint))
      })
      nmOpt.minimize(f, initialSimplex, reportPerf = true)._2 match {
        case Some(perf) =>
          val numIters = perf.stateTrace.size.toDouble
          DenseMatrix(numIters)
        case _ => sys.error("No result found!")
      }
    }): _*)
  }

  def nmConvRate(): Unit = {
    experimentWithResults("Nelder-Mead convergence rate, n=8", "nm-conv-rate.csv") {
      val n = 8
      val nmOpt = new NelderMeadOptimizer(maxObjectiveEvals = Int.MaxValue, maxSteps = Int.MaxValue, tol = 1E-6)
      DenseMatrix.horzcat((for {
        _ <- 0 until 1000
      } yield {
        val initialSimplex = Simplex(Seq.fill(n) {
          val simplexPoint = DenseVector(Uniform(-2D, 2D).sample(), Uniform(-1D, 1D).sample())
          (simplexPoint, f(simplexPoint))
        })
        nmOpt.minimize(f, initialSimplex, reportPerf = true)._2 match {
          case Some(perf) =>
            val numIters = perf.stateTrace.size.toDouble
            DenseMatrix(numIters)
          case _ => sys.error("No result found!")
        }
      }): _*)
    }

    experimentWithResults("Nelder-Mead convergence rate, verying n", "nm-conv-rate-vary-n.csv") {
      val nmOpt = new NelderMeadOptimizer(maxObjectiveEvals = Int.MaxValue, maxSteps = Int.MaxValue, tol = 1E-6)
      DenseMatrix.horzcat((for {
        n <- 3 to 30
        _ <- 0 until 1000
      } yield {
        val initialSimplex = Simplex(Seq.fill(n) {
          val simplexPoint = DenseVector(Uniform(-2D, 2D).sample(), Uniform(-1D, 1D).sample())
          (simplexPoint, f(simplexPoint))
        })
        nmOpt.minimize(f, initialSimplex, reportPerf = true)._2 match {
          case Some(perf) =>
            val numIters = perf.stateTrace.size.toDouble
            DenseMatrix(n.toDouble, numIters)
          case _ => sys.error("No result found!")
        }
      }): _*)
    }
  }

  def nmPerf(): Unit = experimentWithResults(s"Nelder-Mead number simplex points", "nm-vary-n.csv") {
    val nmOpt = new NelderMeadOptimizer(maxObjectiveEvals = 1000, maxSteps = Int.MaxValue, tol = 0D)
    DenseMatrix.horzcat((for {
      n <- 3 to 30
      _ <- 0 until 1000
    } yield {
      val initialSimplex = Simplex(Seq.fill(n) {
        val simplexPoint = DenseVector(Uniform(-2D, 2D).sample(), Uniform(-1D, 1D).sample())
        (simplexPoint, f(simplexPoint))
      })
      nmOpt.minimize(f, initialSimplex, reportPerf = true)._2 match {
        case Some(perf) =>
          val sStar = perf.stateTrace.last
          val xStar = sStar.points.minBy(_._2)._1
          val distToOpt = xOpts.map(xOpt => norm(xStar.toDenseVector - xOpt)).min
          val distInObj = norm(f(xStar) - fOpt)
          val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))

          DenseMatrix(n.toDouble, distToOpt, distInObj, if (closestToGlobal) 1D else 0D)
        case _ => sys.error("No result found!")
      }
    }): _*)
  }

  def nmExample(): Unit = {
    println(s"===Optimizing 6HCF using Nedler-Mead===")
    val nmOpt = new NelderMeadOptimizer(maxObjectiveEvals = 1000, maxSteps = Int.MaxValue, tol = 0D)
    val initialSimplex = Simplex(Seq.fill(8) {
      val simplexPoint = DenseVector(Uniform(-2D, 2D).sample(), Uniform(-1D, 1D).sample())
      (simplexPoint, f(simplexPoint))
    })
    nmOpt.minimize(f, initialSimplex, reportPerf = true) match {
      case (_, Some(perf)) =>
        val sStar = perf.stateTrace.last
        val xStar = sStar.points.minBy(_._2)._1
        val fStar = f(xStar)

        // columns = (x1,y1,x2,y2,...), rows = iterations
        val stateTrace = DenseMatrix.vertcat(perf.stateTrace.map { simplex =>
          DenseMatrix(simplex.points.flatMap(_._1.toArray))
        }: _*)
        val stateTraceFile = new File("results/nm-stateTrace.csv")
        csvwrite(stateTraceFile, stateTrace)
        println(s"Wrote stateTrace to $stateTraceFile")

        // objective value at simplex centroid
        val objTrace = DenseMatrix(perf.stateTrace.map { s =>
          val candidates = s.points.map(_._1)
          f(candidates.reduce(_+_) / candidates.size.toDouble)
        }: _*)
        val objTraceFile = new File("results/nm-objTrace.csv")
        csvwrite(objTraceFile, objTrace)
        println(s"Wrote objTrace to $objTraceFile")

        println(s"xStar: $xStar\n" +
          s"fStar:$fStar\n" +
          s"numIters:${perf.stateTrace.length}\n" +
          s"numObjEval: ${perf.numObjEval}\n" +
          s"numGradEval:${perf.numGradEval}")
      case _ => sys.error("No result found!")
    }
  }

  def gaExample(): Unit = {
    println(s"===Optimizing 6HCF using GA===")
    val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxSteps = Int.MaxValue)
    val popSize = 20
    val eliteCount = 2
    val xoverFrac = 0.8
    ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, Some(seed)) match {
      case (_, Some(perf)) =>
        val xstar = perf.stateTrace.last.population.minBy(_._2)._1
        val fstar = f(xstar)

        // columns = (x1,y1,x2,y2,...), rows = iterations
        val stateTrace = DenseMatrix.vertcat(perf.stateTrace.map { gen =>
          DenseMatrix(gen.population.flatMap(_._1.toArray))
        }: _*)
        val stateTraceFile = new File("results/ga-stateTrace.csv")
        csvwrite(stateTraceFile, stateTrace)
        println(s"Wrote stateTrace to $stateTraceFile")

        // mean population objective value
        val meanObjTrace = DenseMatrix(perf.stateTrace.map { gen =>
          gen.population.map(_._2).sum / gen.population.size.toDouble
        }: _*)
        val minObjTrace = DenseMatrix(perf.stateTrace.map { gen =>
          gen.population.map(_._2).min
        }: _*)
        val objTraceFile = new File("results/ga-objTrace.csv")
        csvwrite(objTraceFile, DenseMatrix.horzcat(meanObjTrace, minObjTrace))
        println(s"Wrote objTraces to $objTraceFile")

        println(s"popSize: $popSize\n" +
          s"eliteCount: $eliteCount\n" +
          s"xstar:$xstar\n" +
          s"fstar:$fstar\n" +
          s"numSteps:${perf.stateTrace.length}\n" +
          s"fEval:${perf.numObjEval}\n" +
          s"dfEval:${perf.numGradEval}")
      case _ => sys.error("No result found!")
    }
  }

  def gaObjEvalEff(): Unit = experimentWithResults("GA obj eval efficiency", "ga-obj-eval-eff.csv"){
    val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxSteps = Int.MaxValue)
    val popSize = 20
    val eliteCount = 2
    val xoverFrac = 0.8
    DenseMatrix.horzcat((for {
      _ <- 0 until 1000
    } yield {
      ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, Some(seed)) match {
        case (_, Some(perf)) =>
          val numGens = perf.stateTrace.size
          DenseMatrix(numGens.toDouble)
        case _ => sys.error("No result found!")
      }
    }): _*)
  }

  def gaPerfPopSize(): Unit = experimentWithResults("GA population size", "ga-pop-size.csv") {
    val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxSteps = Int.MaxValue)
    val popSize = 20
    val eliteCount = 2
    val xoverFrac = 0.8
    DenseMatrix.horzcat((for {
      popSize <- 3 to 50
      _ <- 0 until 1000
    } yield {
      ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, Some(seed)) match {
        case (_, Some(perf)) =>
          val (xStar, fStar) = perf.stateTrace.last.population.minBy(_._2)
          val distInObj = norm(fStar - fOpt)
          val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))

          DenseMatrix(popSize.toDouble, distInObj, if (closestToGlobal) 1D else 0D)
        case _ => sys.error("No result found!")
      }
    }): _*)
  }

  def gaPerfEliteCount(): Unit = experimentWithResults("GA elite count", "ga-elite-count.csv") {
    val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxSteps = Int.MaxValue)
    val popSize = 20
    val eliteCount = 2
    val xoverFrac = 0.8
    DenseMatrix.horzcat((for {
      eliteCount <- 0 until 20
      _ <- 0 until 1000
    } yield {
      ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, Some(seed)) match {
        case (_, Some(perf)) =>
          val (xStar, fStar) = perf.stateTrace.last.population.minBy(_._2)
          val distInObj = norm(fStar - fOpt)
          val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))

          DenseMatrix(eliteCount.toDouble, distInObj, if (closestToGlobal) 1D else 0D)
        case _ => sys.error("No result found!")
      }
    }): _*)
  }

  def gaPerfXoverFrac(): Unit = experimentWithResults("GA xover frac", "ga-xover-frac.csv") {
    val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxSteps = Int.MaxValue)
    val popSize = 20
    val eliteCount = 2
    val xoverFrac = 0.8
    DenseMatrix.horzcat((for {
      xoverFrac <- 0.0 to 1.0 by 0.05
      _ <- 0 until 1000
    } yield {
      ga.minimize(f, lb, ub, popSize, StochasticUniversalSampling, eliteCount, xoverFrac, Some(seed)) match {
        case (_, Some(perf)) =>
          val (xStar, fStar) = perf.stateTrace.last.population.minBy(_._2)
          val distInObj = norm(fStar - fOpt)
          val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))

          DenseMatrix(xoverFrac.toDouble, distInObj, if (closestToGlobal) 1D else 0D)
        case _ => sys.error("No result found!")
      }
    }): _*)
  }

  def gaPerfSelection(): Unit = {
    experimentWithResults("GA selection schemes: non-tournament", "ga-selection.csv") {
      val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxSteps = Int.MaxValue)
      val popSize = 20
      val eliteCount = 2
      val xoverFrac = 0.8

      DenseMatrix.horzcat((for {
        scheme <- List(FitnessProportionateSelection, StochasticUniversalSampling)
        _ <- 0 until 1000
      } yield {
        ga.minimize(f, lb, ub, popSize, scheme, eliteCount, xoverFrac, Some(seed)) match {
          case (_, Some(perf)) =>
            val (xStar, fStar) = perf.stateTrace.last.population.minBy(_._2)
            val distInObj = norm(fStar - fOpt)
            val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))

            DenseMatrix(distInObj, if (closestToGlobal) 1D else 0D)
          case _ => sys.error("No result found!")
        }
      }): _*)
    }

    experimentWithResults("GA selection schemes: tournament", "ga-selection-tournament.csv") {
      val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxSteps = Int.MaxValue)
      val popSize = 20
      val eliteCount = 2
      val xoverFrac = 0.8

      DenseMatrix.horzcat((for {
        tournamentProb <- 0.05 to 1.0 by 0.05
        _ <- 0 until 1000
      } yield {
        ga.minimize(f, lb, ub, popSize, TournamentSelection(tournamentProb), eliteCount, xoverFrac, Some(seed)) match {
          case (_, Some(perf)) =>
            val (xStar, fStar) = perf.stateTrace.last.population.minBy(_._2)
            val distInObj = norm(fStar - fOpt)
            val closestToGlobal = xOpts.contains(localMinima.minBy(xMin => norm(xMin - xStar)))

            DenseMatrix(tournamentProb.toDouble, distInObj, if (closestToGlobal) 1D else 0D)
          case _ => sys.error("No result found!")
        }
      }): _*)
    }
  }

  def experimentWithResults(
    experimentName: String,
    resultFName: String) = (results: Matrix[Double]) => {
    println(s"=== Performing experiment: $experimentName ===")
    val resultsFile = new File(s"results/$resultFName")
    println(s"Writing results to: $resultsFile")
    csvwrite(resultsFile, results)
  }
}
