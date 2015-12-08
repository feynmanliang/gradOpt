package com.feynmanliang.gradopt

import java.io.File

import breeze.linalg._
import breeze.numerics._
import breeze.plot._

// Performance diagnostics for the optimizer
private[gradopt] case class PerfDiagnostics[T](
  xTrace: Seq[(T, Double)],
  numEvalF: Long,
  numEvalDf: Long)

private[gradopt] class FunctionWithCounter[-T,+U](f: T => U) extends Function[T,U] {
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
  import com.feynmanliang.gradopt.GradientAlgorithm._
  import com.feynmanliang.gradopt.LineSearchConfig._

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

    val gradOpt = new GradientOptimizer()
    for (x0 <- List(-5, -1, -0.1, -1E-2, -1E-3, -1E-4, -1E-5, 1E-5, 1E-4, 1E-3, 1E-2, 0.1, 1, 5)) {
      gradOpt.minimize(f, df, x0, SteepestDescent, CubicInterpolation, true) match {
        case (Some(xstar), Some(perf)) =>
          println(f"x0=$x0, xstar=$xstar, numEvalF=${perf.numEvalF}, numEvalDf=${perf.numEvalDf}")
          println(perf.xTrace.toList.map("%2.4f".format(_)))
        case _ => println(s"No results for x0=$x0!!!")
      }
    }
  }

  def q3(showPlot: Boolean = false): Unit = {
    val gradOpt = new GradientOptimizer(maxSteps=101, tol=1E-4)
    for {
      fname <- List("A10.csv", "A100.csv", "A1000.csv", "B10.csv", "B100.csv", "B1000.csv")
    } {
      val A: DenseMatrix[Double] = csvread(new File(getClass.getResource("/" + fname).getFile()))
      assert(A.rows == A.cols, "A must be symmetric")
      val n: Int = A.cols
      val b: DenseVector[Double] = 2D * (DenseVector.rand(n) - DenseVector.fill(n){0.5})

      println(s"$fname")
      gradOpt.minQuadraticForm(A, b, DenseVector.zeros(n), SteepestDescent, Exact, true) match {
        case (res, Some(perf)) =>
          println(s"$res, ${perf.xTrace.takeRight(2)}, ${perf.xTrace.length}, ${perf.numEvalF}, ${perf.numEvalDf}")
        case _ => throw new Exception("Minimize failed to return perf diagnostics")
      }
    }
  }

  def main(args: Array[String]) = {
    // q2(showPlot = false)
    q3(showPlot = false)
  }
}

// vim: set ts=2 sw=2 et sts=2:
