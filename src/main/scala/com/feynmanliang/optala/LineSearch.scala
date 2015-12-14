package com.feynmanliang.optala

import breeze.linalg._
import breeze.numerics._

/**
 * A bracketing interval where f(x + mid'*df) < f(x + lb'*df) and f(x + mid'*df) < f(x + ub'*df),
 * ensuring a minimum is within the bracketed interval.
 */
private[optala] case class BracketInterval(lb: Double, mid: Double, ub: Double) {
  def contains(x: Double): Boolean = lb <= x && ub >= x
  def size: Double = ub - lb
}

object LineSearch {
  /**
  * Brackets a step size `alpha` such that for some value within the bracket
  * the restriction of `f` to the ray `f(x + alpha*p)` is guaranteed to attain
  * a minimum.
  */
  private[optala] def bracket(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x: Vector[Double],
      p: Vector[Double],
      maxBracketIters: Int = 5000): Option[BracketInterval] = {
    val (phi, dPhi) = restrictRay(f, df, x, p)
    val fx0 = phi(0)
    val dfx0 = dPhi(0)

    if (norm(dfx0) == 0.0D) {
      Some(BracketInterval(-1E-2, 0D, 1E-2))
    } else {
      /** A stream of successively expanding brackets until a valid bracket is found */
      def nextBracket(currBracket: BracketInterval): Stream[BracketInterval] = currBracket match {
        case BracketInterval(lb, mid, ub) =>
          val fMid = fx0 // TODO: adapt midpoint
          val flb = phi(lb)
          val fub = phi(ub)
          val newLb = if (fMid < flb) lb else lb - (mid - lb)
          val newUb = if (fMid < fub) ub else ub + (ub - mid)
          currBracket #:: nextBracket(BracketInterval(newLb, mid, newUb))
      }

      val initBracket = BracketInterval(-1E-2D, 0D, 1E-2D)
      nextBracket(initBracket)
        .take(maxBracketIters)
        .find({ case BracketInterval(lb, mid, ub) =>
          val fMid = fx0 // TODO: adapt midpoint
          phi(lb) > fMid && phi(ub) > fMid
        })
    }
  }

  /**
  * Performs a line search for x' = x + a*p within a bracketing interval to determine step size.
  * Returns the value x' which minimizes `f` along the line search. The chosen step size
  * satisfies the Strong Wolfe Conditions.
  */
  def chooseStepSize(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x: Vector[Double],
      p: Vector[Double],
      c1: Double = 1E-4,
      c2: Double = 0.9): Option[Double] = LineSearch.bracket(f, df, x, p) match {
    case _ if norm(p.toDenseVector) < 1E-6 => Some(0D)// degenerate ray direction
    case None => None // unable to bracket
    case Some(bracket) =>
      val aMax: Double = bracket.ub // min guaranteed to be attained by alpha within bracket

      val (phi, dPhi) = restrictRay(f, df, x, p)
      val phiZero = phi(0)
      val dPhiZero = dPhi(0)

      /**
      * Nocedal Algorithm 3.5, finds a step length \alpha while ensures that
      * (aPrev, aCurr) contains a point satisfying the Strong Wolfe Conditions at
      * each iteration.
      */
      def chooseAlpha(aPrev: Double, aCurr: Double, firstIter: Boolean): Option[Double] = {
        val phiPrev = phi(aPrev)
        val phiCurr = phi(aCurr)

        if (phiCurr > phiZero + c1*aCurr*dPhiZero || (phiCurr >= phiPrev && !firstIter)) {
          zoom(aPrev, aCurr)
        } else {
          val dPhiCurr = dPhi(aCurr)
          if (math.abs(dPhiCurr) <= -1*c2 * dPhiZero) {
            Some(aCurr)
          }
          else if (dPhiCurr >= 0) {
            zoom(aCurr, aPrev)
          } else {
            chooseAlpha(aCurr, (aCurr + aMax) / 2D, firstIter=false)
          }
        }
      }

      /**
      * Nocedal Algorithm 3.6, generates \alpha_j between \alpha_{lo} and \alpha_{hi} and replaces
      * one of the two endpoints while ensuring Wolfe conditions hold.
      */
      def zoom(alo: Double, ahi: Double): Option[Double] = {
        assert(!alo.isNaN && !ahi.isNaN)
        interpolate(alo, ahi) match {
          case Some(aCurr) if math.abs(ahi - alo) > 1E-8 =>
            val phiACurr = phi(aCurr)
            if (phiACurr > phiZero + c1 * aCurr * dPhiZero || phiACurr >= phi(alo)) {
              zoom(alo, aCurr)
            } else {
              val dPhiCurr = dPhi(aCurr)
              if (math.abs(dPhiCurr) <= -c2 * dPhiZero) {
                Some(aCurr)
              } else if (dPhiCurr * (ahi - alo) >= 0) {
                zoom(aCurr, alo)
              } else {
                zoom(aCurr, ahi)
              }
            }
          case _ => Some((ahi + alo) / 2D)
        }
      }

      /**
      * Finds the minimizer of the Cubic interpolation of the line search
      * objective \phi(\alpha) between [alpha_{i-1}, alpha_i]. See Nocedal (3.59).
      **/
      def interpolate(prev: Double, curr: Double): Option[Double] = {
        val d1 = dPhi(prev) + dPhi(curr) - 3D * (phi(prev) - phi(curr)) / (prev - curr)
        val d2 = signum(curr - prev) * sqrt(pow(d1,2) - dPhi(prev) * dPhi(curr))
        val res = curr - (curr - prev) * (dPhi(curr) + d2 - d1) / (dPhi(curr) - dPhi(prev) + 2D*d2)

        if (!res.isNaN) Some(res) else None
      }

      chooseAlpha(0, aMax / 2D, firstIter=true)
  }

  /** Restricts a vector function `f` with derivative `df` along ray `f(x + alpha * p)` */
  private def restrictRay(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x: Vector[Double],
      p: Vector[Double]): (Double => Double, Double => Double) = {
    (alpha => f(x + alpha * p), alpha => df(x + alpha * p) dot p)
  }


  /**
  * Computes the exact step size alpha required to minimize a quadratic form.
  * See Nocedal (3.55).
  */
  def exactLineSearch(
      A: Matrix[Double],
      df: Vector[Double],
      x: Vector[Double],
      p: Vector[Double]): Option[Double] = {
    if (norm(p.toDenseVector) == 0D) Some(0D) // degenerate ray
    else {
      val num = -(df.t * p)
      val denom = p.t * (A * p) // assumes A is PSD
      denom match {
        case 0 => None
        case _ => Some(num / denom)
      }
    }
  }
}


// vim: set ts=2 sw=2 et sts=2:
