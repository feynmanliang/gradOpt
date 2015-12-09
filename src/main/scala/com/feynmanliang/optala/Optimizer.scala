package com.feynmanliang.optala

import java.io.File

import breeze.linalg._
import breeze.numerics._
import breeze.plot._

// Performance diagnostics for the optimizer
private[optala] case class PerfDiagnostics[T](
  xTrace: Seq[(T, Double)],
  numEvalF: Long,
  numEvalDf: Long)

private[optala] class FunctionWithCounter[-T,+U](f: T => U) extends Function[T,U] {
  var numCalls: Int = 0
  override def apply(t: T): U = {
    numCalls += 1
    f(t)
  }
}

/**
* Client implementing example code.
* TODO: move to separate client
*/
object Optimizer {
  import com.feynmanliang.optala.GradientAlgorithm._
  import com.feynmanliang.optala.LineSearchConfig._

  def q2(showPlot: Boolean = false): Unit = {
    val f = (x: Double) => pow(x,4) * cos(pow(x,-1)) + 2D * pow(x,4)
    val df = (x: Double) => 8D * pow(x,3) + 4D * pow(x, 3) * cos(pow(x,-1)) - pow(x, 4) * sin(pow(x,-1))

    if (showPlot) {
      val fig = Figure()
      val x = linspace(-.1,0.1)
      fig.subplot(2,1,0) += plot(x, x.map(f))
      fig.subplot(2,1,1) += plot(x, x.map(df))
      fig.saveas("lines.png") // save current figure as a .png, eps and pdf also supported
    }

    val optala = new GradientOptimizer(maxSteps=5000, tol=1E-6)
    for (x0 <- List(-5, -1, -0.1, -1E-2, -1E-3, -1E-4, -1E-5, 1E-5, 1E-4, 1E-3, 1E-2, 0.1, 1, 5)) {
      optala.minimize(f, df, x0, SteepestDescent, CubicInterpolation, true) match {
        case (_, Some(perf)) =>
          val (xstar, fstar) = perf.xTrace.last
          println(s"x0:$x0,xstar:$xstar,fstar:$fstar,normGrad:${perf.xTrace.last._2},numSteps:${perf.xTrace.length},fEval:${perf.numEvalF},dfEval:${perf.numEvalDf}")
        case _ => println(s"No results for x0=$x0!!!")
      }
    }
  }

  def q4(showPlot: Boolean = false): Unit = {
    val f: Vector[Double] => Double = v => pow(1D - v(0), 2) + 100D * pow(v(1) - pow(v(0), 2),2)
    val df: Vector[Double] => Vector[Double] = v => {
      DenseVector(
        -2D*(1 - v(0)) - 400D * v(0) * (-pow(v(0), 2) + v(1)),
        200D * (-pow(v(0), 2) + v(1))
      )
    }
    val xOpt = (-1D, 1D)
    val x0 = DenseVector(-3D, -4D)

    val optala = new GradientOptimizer(maxSteps=5000, tol=1E-6)
    for (algo <- List(SteepestDescent, ConjugateGradient)) {
      println(s"Optimizing Rosenbrock function using $algo")
      optala.minimize(f, df, x0, algo, CubicInterpolation, true) match {
        case (_, Some(perf)) =>
          val (xstar, fstar) = perf.xTrace.last
          println(s"x0:$x0,xstar:$xstar,fstar:$fstar,normGrad:${perf.xTrace.last._2},numSteps:${perf.xTrace.length},fEval:${perf.numEvalF},dfEval:${perf.numEvalDf}")
        case _ => println(s"No results for x0=$x0!!!")
      }
    }

    val nmOpt = new NelderMeadOptimizer(maxSteps=5000, tol=1E-10)
    println(s"Optimizing Rosenbrock function using Nedler-Mead")
    nmOpt.minimize(f, 2, 8, true) match {
      case (_, Some(perf)) =>
        val (sstar, fstar) = perf.xTrace.last
        val xstar = sstar.points.map(_._1).reduce(_+_)/ (1D*sstar.points.size)
        println(s"x0:$x0,xstar:$xstar,fstar:$fstar,normGrad:${perf.xTrace.last._2},numSteps:${perf.xTrace.length},fEval:${perf.numEvalF},dfEval:${perf.numEvalDf}")
      case _ => println(s"No results for x0=$x0!!!")
    }
  }

  def q56(showPlot: Boolean = false): Unit = {
    val optala = new GradientOptimizer(maxSteps=101, tol=1E-4)
    for {
      lsAlgo <- List(Exact, CubicInterpolation);
      fname <- List("A10.csv", "A100.csv", "A1000.csv", "B10.csv", "B100.csv", "B1000.csv")
      optAlgo <- List(SteepestDescent, ConjugateGradient)
    } {
      val A: DenseMatrix[Double] = csvread(new File(getClass.getResource("/" + fname).getFile()))
      assert(A.rows == A.cols, "A must be symmetric")
      val n: Int = A.cols
      val b: DenseVector[Double] = 2D * (DenseVector.rand(n) - DenseVector.fill(n){0.5})

      println(s"lsAlgo:$lsAlgo,fname:$fname,optAlgo:$optAlgo")
      optala.minQuadraticForm(A, b, DenseVector.zeros(n), optAlgo, lsAlgo, true) match {
        case (res, Some(perf)) =>
          println(s"normGrad:${perf.xTrace.last._2},numSteps:${perf.xTrace.length},fEval:${perf.numEvalF},dfEval:${perf.numEvalDf}")
        case _ => throw new Exception("Minimize failed to return perf diagnostics")
      }
    }
  }

  def nmVsGa(): Unit = {
    val f: Vector[Double] => Double = v => {
      val x = v(0)
      val y = v(1)
      (4D - 2.1D*pow(x,2) + (1D/3D)*pow(x,4))*pow(x,2) + x*y + (4D*pow(y,2) - 4D)*pow(y,2)
    }
    val lb = DenseVector(-2D, -1D)
    val ub = DenseVector(2D, 1D)


    println(s"Optimizing 6HCF using Nedler-Mead")
    val nmOpt = new NelderMeadOptimizer(maxSteps=1000, tol=1E-10)
    val x0 = DenseVector(-1D, -1D)
    nmOpt.minimize(f, 2, 8, true) match {
      case (_, Some(perf)) =>
        val (sstar, fstar) = perf.xTrace.last
        val xstar = sstar.points.map(_._1).reduce(_+_)/ (1D*sstar.points.size)
        println(s"x0:$x0,xstar:$xstar,fstar:$fstar,normGrad:${perf.xTrace.last._2},numSteps:${perf.xTrace.length},fEval:${perf.numEvalF},dfEval:${perf.numEvalDf}")
      case _ => println(s"No results for x0=$x0!!!")
    }

    println(s"Optimizing 6HCF using GA")
    val ga = new GeneticAlgorithm(maxSteps=1000)
    val popSize = 20
    val eliteCount = 2
    val xoverFrac = 0.8
    val seed = 42
    ga.minimize(f, lb, ub, popSize, eliteCount, xoverFrac, Some(seed)) match {
      case (_, Some(perf)) =>
        val (xstar, fstar) = perf.xTrace.last._1.population.minBy(_._2)
        println(s"popSize:$popSize,xstar:$xstar,fstar:$fstar,numSteps:${perf.xTrace.length},fEval:${perf.numEvalF},dfEval:${perf.numEvalDf}")
      case _ => println(s"No results for x0=$x0!!!")
    }
  }

  def main(args: Array[String]) = {
    //q2(showPlot = false)
    //q4(showPlot = false)
    //q56(showPlot = false)
    nmVsGa()
  }
}

// vim: set ts=2 sw=2 et sts=2:
