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
//    runNelderMead

    println(s"===Exploring Nelder-Mead number simplex points ==")
    val nmOpt = new NelderMeadOptimizer(maxObjectiveEvals = 1000, maxSteps = Int.MaxValue, tol = 0D)
    val results = DenseMatrix.vertcat((for {
      n <- 3 until 20
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
          val closestToGlobal = xOpts.contains(localMinima .minBy(xMin => norm(xMin - xStar)))

          DenseMatrix(n.toDouble, distToOpt, distInObj, if (closestToGlobal) 1D else 0D)
        case _ => sys.error("No result found!")
      }
    }): _*)
    val resultFile = new File("results/nm-vary-n.csv")
    csvwrite(resultFile, results)
    println(s"Wrote results to $resultFile")
//    runGA
  }

  def runNelderMead(): Unit = {
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

        // objective value at best point on simplex
        val objTrace = DenseMatrix(perf.stateTrace.map(_.points.map(_._2).min): _*)
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

  def runGA(): Unit = {
    println(s"===Optimizing 6HCF using GA===")
    val ga = new GeneticAlgorithm(maxObjectiveEvals = 1000, maxSteps = Int.MaxValue)
    val popSize = 20
    val eliteCount = 2
    val xoverFrac = 0.8
    ga.minimize(f, lb, ub, popSize, TournamentSelection(0.5), eliteCount, xoverFrac, Some(seed)) match {
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

        // objective value at best point on simplex
        val objTrace = DenseMatrix(perf.stateTrace.map(_.population.map(_._2).min): _*)
        val objTraceFile = new File("results/ga-objTrace.csv")
        csvwrite(objTraceFile, objTrace)
        println(s"Wrote objTrace to $objTraceFile")

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

}
