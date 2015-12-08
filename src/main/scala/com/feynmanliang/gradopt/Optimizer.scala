package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._
import breeze.plot._


// A bracketing interval where f(x + mid'*df) < f(x + lb'*df) and f(x + mid'*df) < f(x + ub'*df),
// guaranteeing a minimum
private[gradopt] case class BracketInterval(lb: Double, mid: Double, ub: Double) {
  def contains(x: Double): Boolean = lb <= x && ub >= x
  def size: Double = ub - lb
}

// Performance diagnostics for the optimizer
private[gradopt] case class PerfDiagnostics(
  xTrace: Seq[Vector[Double]],
  numEvalF: Long,
  numEvalDf: Long)

// TODO: these should be path-dependent to restrict to optimizer instance
private[gradopt] class FunctionWithCounter[-T,+U](f: T => U) extends Function[T,U] {
  var numCalls: Int = 0
  override def apply(t: T): U = {
    numCalls += 1
    f(t)
  }
}


class Optimizer(
  var maxStepIters: Int = 5000,
  var tol: Double = 1E-8) {

  // Overload to permit scalar valued functions
  def minimize(
      f: Double => Double,
      df: Double => Double,
      x0: Double,
      reportPerf: Boolean): (Option[Vector[Double]], Option[PerfDiagnostics]) = {
    val vecF: Vector[Double] => Double = v => {
      require(v.size == 1, s"vectorized f expected dimension 1 input but got ${v.size}")
      f(v(0))
    }
    val vecDf: Vector[Double] => Vector[Double] = v => {
      require(v.size == 1, s"vectorized f expected dimension 1 input but got ${v.size}")
      DenseVector(df(v(0)))
    }
    minimize(vecF, vecDf, DenseVector(x0), reportPerf)
  }

  /**
  * Minimize a convex function `f` with derivative `df` and initial
  * guess `x0`. Uses steepest-descent.
  */
  def minimize(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x0: Vector[Double],
      reportPerf: Boolean): (Option[Vector[Double]], Option[PerfDiagnostics]) = {

    val fCnt = new FunctionWithCounter(f)
    val dfCnt = new FunctionWithCounter(df)

    def improve(x: Vector[Double]): Stream[Vector[Double]] = {
      LineSearch.chooseStepSize(fCnt, -dfCnt(x), dfCnt, x) match {
        case Some(alpha) => {
          val xnew = x - alpha * dfCnt(x)
          x #:: improve(xnew)
        }
        case None => x #:: Stream.Empty
      }
    }

    val xValues = improve(x0).iterator
    val xTrace = xValues
      .take(maxStepIters) // limit max iterations
      .takeWhile((x:Vector[Double]) => norm(df(x).toDenseVector) >= tol) // termination condition based on norm(grad)

    if (reportPerf) {
      val trace = xTrace.toList :+ xValues.next() // Force the lazy Stream and append the last value
      val res = if (trace.length == maxStepIters) None else Some(trace.last)
      val perf = PerfDiagnostics(trace, fCnt.numCalls, dfCnt.numCalls)
      (res, Some(perf))
    } else {
      val res = xValues.find((x:Vector[Double]) => norm(df(x).toDenseVector) < tol)
      (res, None)
    }
  }
}


/**
* Client for Optimizer implementing coursework questions
*/
object Optimizer {

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

    val opt = new Optimizer()
    for (x0 <- List(-5, -1, -0.1, -1E-2, -1E-3, -1E-4, -1E-5, 1E-5, 1E-4, 1E-3, 1E-2, 0.1, 1, 5)) {
      opt.minimize(f, df, x0, true) match {
        case (Some(xstar), Some(perf)) =>
          println(f"x0=$x0, xstar=$xstar, numEvalF=${perf.numEvalF}, numEvalDf=${perf.numEvalDf}")
          println(perf.xTrace.toList.map("%2.4f".format(_)))
        case _ => println(s"No results for x0=$x0!!!")
      }
    }
  }
  def q3(showPlot: Boolean = false): Unit = ???
  def main(args: Array[String]) = {
    // q2(showPlot = false)
    q3(showPlot = false)
  }
}

// vim: set ts=2 sw=2 et sts=2:
