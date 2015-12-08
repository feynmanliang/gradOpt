package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._
import breeze.plot._

object LineSearch {
  /**
  * Brackets the minimum of a function `f`. This function uses `x0` as the
  * midpoint and `df` as the line around which to find bracket bounds.
  * TODO: better initialization
  * TODO: update the midpoint to be something besides 0
  */
  private[gradopt] def bracket(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x0: Vector[Double],
      maxBracketIters: Int = 5000): Option[BracketInterval] = {
    val fx0 = f(x0)
    val dfx0 = df(x0)
    if (norm(dfx0.toDenseVector) < 1E-6) return Some(BracketInterval(-1E-6, 0D, 1E-2))

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
  */
  def chooseStepSize(
      f: Vector[Double] => Double,
      p: Vector[Double],
      df: Vector[Double] => Vector[Double],
      x: Vector[Double],
      c1: Double = 1E-4,
      c2: Double = 0.9): Option[Double] = LineSearch.bracket(f, df, x)  match {
    case None => None
    case Some(bracket) => {
      val aMax: Double = 2 // max step length, TODO: use bracket

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
        assert(!alo.isNaN && !ahi.isNaN)
        val aCurr = interpolate(alo, ahi)
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

      Some(chooseAlpha(0, aMax / 2D, true))
    }
  }
}


// vim: set ts=2 sw=2 et sts=2:
