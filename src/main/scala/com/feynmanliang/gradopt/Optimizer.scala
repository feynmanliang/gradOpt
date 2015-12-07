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

class Optimizer(
  var maxBracketIters: Int = 5000,
  var maxStepIters: Int = 5000,
  var tol: Double = 1E-8
) {
  private[gradopt] class FunctionWithCounter[-T,+U](f: T => U) extends Function[T,U] {
    var numCalls: Int = 0
    override def apply(t: T): U = {
      numCalls += 1
      f(t)
    }
  }

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
  * guess `x0`.
  */
  def minimize(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x0: Vector[Double],
      reportPerf: Boolean): (Option[Vector[Double]], Option[PerfDiagnostics]) = {

    val fCnt = new FunctionWithCounter(f)
    val dfCnt = new FunctionWithCounter(df)

    // Stream of x values returned by bracket/line search algorithm
    def improve(x: Vector[Double]): Stream[Vector[Double]] = {
      bracket(fCnt, dfCnt, x) match {
        case Some(bracket) => {
          val xnew = lineSearch(fCnt, dfCnt(x), x, bracket)
          x #:: improve(xnew)
        }
        case None => x #:: Stream.Empty
      }
    }

    val xValues = improve(x0)
      .take(maxStepIters) // limit max iterations
      .takeWhile((x:Vector[Double]) => norm(df(x).toDenseVector) >= tol) // termination condition based on norm(grad)

    if (reportPerf) {
      val trace = xValues.toSeq
      val res = if (trace.length == maxStepIters) None else Some(trace.last)
      val perf = PerfDiagnostics(trace, fCnt.numCalls, dfCnt.numCalls)
      (res, Some(perf))
    } else {
      val res = xValues.find((x:Vector[Double]) => norm(df(x).toDenseVector) >= tol)
      (res, None)
    }
  }

  /**
  * Brackets the minimum of a function `f`. This function uses `x0` as the
  * midpoint and `df` as the line around which to find bracket bounds.
  * TODO: better initialization
  * TODO: update the midpoint to be something besides 0
  */
  private[gradopt] def bracket(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x0: Vector[Double]): Option[BracketInterval] = {
    val fx0 = f(x0)
    val dfx0 = df(x0)
    if (norm(dfx0.toDenseVector) < tol) return Some(BracketInterval(-1E-6, 0D, 1E-6))

    def nextBracket(currBracket: BracketInterval): Stream[BracketInterval] = currBracket match {
      case BracketInterval(lb, mid, ub) => {
        val fMid = fx0 // TODO: adapt midpoint
        val flb = f(x0 - lb * dfx0)
        val fub = f(x0 - ub * dfx0)
        val newLb = if (fMid < flb) lb else lb - (mid - lb)
        val newUb = if (fMid < fub) ub else ub + (ub - mid)
        currBracket #:: nextBracket(BracketInterval(newLb, mid, newUb))
      }
    }

    val initBracket = BracketInterval(-0.1D, 0D, 0.1D)
    nextBracket(initBracket)
      .take(maxBracketIters)
      .find(_ match {
        case BracketInterval(lb, mid, ub) => {
          val fMid = fx0 // TODO: adapt midpoint
          f(x0 - lb * dfx0) > fMid && f(x0 - ub * dfx0) > fMid
        }
      })
  }

  /**
  * Performs a line search for x' = x + a*p within a bracketing interval to determine step size.
  * Returns the value x' which minimizes `f` along the line search.
  * This method linearly interpolates the bracket interval and chooses the minimizer of f.
  * TODO: bisection search the candidates
  */
  private[gradopt] def lineSearch(
      f: Vector[Double] => Double,
      dfx: Vector[Double],
      x: Vector[Double],
      bracket: BracketInterval): Vector[Double] = {
    val numPoints = 100D // TODO: increase this if bracketing doesn't improve
    val candidates = (0D +: (bracket.lb to bracket.ub by bracket.size/numPoints))
      .map(p => x - p * dfx)
    candidates.minBy(f.apply)
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
  def main(args: Array[String]) = {
    q2()
  }
}

// vim: set ts=2 sw=2 et sts=2:
