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
  * guess `x0`. Uses steepest-descent.
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
          val xnew = x - lineSearch(fCnt, dfCnt, x, bracket) * dfCnt(x)
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
    if (norm(dfx0.toDenseVector) < tol) return Some(BracketInterval(-1E-6, 0D, 1E-2))

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
  * Returns the value x' which minimizes `f` along the line search. The chosen step size
  * satisfies the Strong Wolfe Conditions.
  * TODO: localize all bracketing into linesearch
  */
  def lineSearch(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x: Vector[Double],
      bracket: BracketInterval): Double = lineSearch(f, -df(x), df, x)

  def lineSearch(
      f: Vector[Double] => Double,
      p: Vector[Double],
      df: Vector[Double] => Vector[Double],
      x: Vector[Double]): Double = {
    val c1: Double = 1E-4
    val c2: Double = 0.9

    val bracket = this.bracket(f, df, x).get // TODO: terminate search if fail
    val aMax: Double = 2 // max step length // TODO: why does this fail when set to bracket.ub

    val phi: Double => Double = alpha => f(x + alpha * p)
    val dPhi: Double => Double = alpha => df(x + alpha * p) dot (p)


    val phiZero = phi(0)
    val dPhiZero = dPhi(0)

    /**
    * Nocedal Algorithm 3.5, finds a step length \alpha while ensures that
    * (aPrev, aCurr) contains a point satisfying the Strong Wolfe Conditions at
    * each iteration.
    */
    def chooseAlpha(aPrev: Double, aCurr: Double, firstIter: Boolean): Double = {
      val phiPrev = phi(aPrev)
      val phiCurr = phi(aCurr)

      if (phiCurr > phiZero + c1*aCurr*dPhiZero || (phiCurr >= phiPrev && !firstIter)) {
        zoom(aPrev, aCurr)
      } else {
        val dPhiCurr = dPhi(aCurr)
        if (math.abs(dPhiCurr) <= -1*c2 * dPhiZero) {
          aCurr
        }
        else if (dPhiCurr >= 0) {
          zoom(aCurr, aPrev)
        } else {
          chooseAlpha(aCurr, (aCurr + aMax) / 2D, false)
        }
      }
    }

    /**
    * Nocedal Algorithm 3.6, generates \alpha_j between \alpha_{lo} and \alpha_{hi} and replaces
    * one of the two endpoints while ensuring Wolfe conditions hold.
    */
    def zoom(alo: Double, ahi: Double): Double = {
      //println(s"zoom: $alo, $ahi")
      assert(!alo.isNaN && !ahi.isNaN)
      val aCurr = interpolate(alo, ahi)
      //println(s"zoom: $aCurr")
      if (phi(aCurr) > phiZero + c1 * aCurr * dPhiZero || phi(aCurr) >= phi(alo)) {
        zoom(alo, aCurr)
      } else {
        val dPhiCurr = dPhi(aCurr)
        if (math.abs(dPhiCurr) <= -c2 * dPhiZero) {
          aCurr
        } else if (dPhiCurr * (ahi - alo) >= 0) {
          zoom(aCurr, alo)
        } else {
          zoom(aCurr, ahi)
        }
      }
    }

    /**
    * Finds the minimizer of the Cubic interpolation of the line search
    * objective \phi(\alpha) between [alpha_{i-1}, alpha_i]. See Nocedal (3.59).
    **/
    def interpolate(prev: Double, curr: Double): Double = {
      val d1 = dPhi(prev) + dPhi(curr) - 3D * (phi(prev) - phi(curr)) / (prev - curr)
      val d2 = signum(curr - prev) * sqrt(pow(d1,2) - dPhi(prev) * dPhi(curr))
      curr - (curr - prev) * (dPhi(curr) + d2 - d1) / (dPhi(curr) - dPhi(prev) + 2D*d2)
    }

    chooseAlpha(0, aMax / 2D, true)
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
