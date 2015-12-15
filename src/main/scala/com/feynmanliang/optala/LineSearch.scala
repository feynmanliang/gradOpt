package com.feynmanliang.optala

import breeze.linalg._
import breeze.numerics._

/**
  * A bracketing interval where f(x + mid'*df) < f(x + lb'*df) and f(x + mid'*df) < f(x + ub'*df),
  * ensuring a minimum is within the bracketed interval.
  */
private[optala] case class BracketInterval(
    lb: Double,
    mid: Double,
    ub: Double,
    fLb: Double,
    fMid: Double,
    fUb: Double) {
  require(lb <= mid && mid <= ub, s"bracket did not satisfy $lb <= $mid <= $ub")
  def contains(x: Double): Boolean = lb <= x && ub >= x
  def size: Double = ub - lb
  def bracketsMin: Boolean = fLb >= fMid && fMid <= fUb
}

object BracketInterval {
  def apply(phi: Double => Double, lb: Double, mid: Double, ub: Double): BracketInterval =
    BracketInterval(lb, mid, ub, phi(lb), phi(mid), phi(ub))
}

object LineSearch {
  private val GOLD = 1.61803398875
  private val EPS_MIN = 1E-16

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
      initialBracketRange: Double = 1D,
      maxBracketIters: Int = 5000): Option[BracketInterval] = {
    val (phi, dPhi) = restrictRay(f, df, x, p)
//    val initBracket = BracketInterval(phi, -1E-8, 0D, 1E-8)
    val initBracket = BracketInterval(phi, 0, initialBracketRange, (1 + GOLD) * initialBracketRange)
    if (norm(dPhi(0)) == 0.0D) {
      Some(initBracket)
    } else {
      /** A stream of successively expanding brackets until a valid bracket is found */
      def nextBracket(currBracket: BracketInterval): Stream[BracketInterval] = currBracket match {
        case BracketInterval(lb, mid, ub, fLb, fMid, fUb) =>
          val newLb = if (fMid < fLb) lb else lb + (lb - mid)
          val newUb = if (fMid < fUb) ub else ub + (ub - mid)
          currBracket #:: nextBracket(BracketInterval(phi, newLb, mid, newUb))
      }
      nextBracket(initBracket)
        .take(maxBracketIters)
        .find(_.bracketsMin)
    }
  }

  /**
  * Performs a line search for x' = x + a*p within a bracketing interval to determine step size.
  * Returns the value `x` which minimizes `f` along the line search. The chosen step size
  * satisfies the Strong Wolfe Conditions.
  */
  def chooseStepSize(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x: Vector[Double],
      p: Vector[Double],
      c1: Double = 1E-4,
      c2: Double = 0.9,
      aMax: Double = 50D): Option[Double] = LineSearch.bracket(f, df, x, p) match {
    case _ if norm(p.toDenseVector) < EPS_MIN => Some(0D) // degenerate ray direction
    case None => None // unable to bracket a minimum
    case Some(bracket) =>
      val aMax = bracket.size
      val initialBracketRatio = 1E-8 // ratio of aMax first Wolfe Condition bracket should be

//      val (phi, dPhi) = restrictRay(f, df, x + bracket.lb*p, p)
      val (phi, dPhi) = restrictRay(f, df, x, p)
      val phiZero = phi(0)
      val dPhiZero = dPhi(0)

      /**
        * Nocedal Algorithm 3.5, finds a step length alpha while ensures that
        * (aPrev, aCurr) contains a point satisfying the Strong Wolfe Conditions at
        * each iteration.
        */
      def findAlpha(aPrev: Double, phiPrev: Double, aCurr: Double, firstIter: Boolean): Option[Double] = {
        if (aCurr > aMax) {
          Some(aMax)
        } else {
          val phiCurr = phi(aCurr)
          if (phiCurr > phiZero + c1 * aCurr * dPhiZero || (phiCurr >= phiPrev && !firstIter)) {
            zoom(aPrev, aCurr)
          } else {
            val dPhiCurr = dPhi(aCurr)
            if (math.abs(dPhiCurr) <= -1 * c2 * dPhiZero) {
              Some(aCurr)
            } else if (dPhiCurr >= 0) {
              zoom(aCurr, aPrev)
            } else {
              //            findAlpha(aCurr, phiCurr, (aCurr + aMax) / 2D, firstIter = false)
              findAlpha(aCurr, phiCurr, 2 * aCurr, firstIter = false)
            }
          }
        }
      }

      /**
        * Nocedal Algorithm 3.6, generates \alpha_j between \alpha_{lo} and \alpha_{hi} and replaces
        * one of the two endpoints while ensuring Wolfe conditions hold.
        */
      def zoom(aLo: Double, aHi: Double): Option[Double] = {
        assert(!aLo.isNaN && !aHi.isNaN)
        interpolate(aLo, aHi) match { // cubic interpolation
//        Some((aLo + aHi) / 2) match { // bisection search
          case Some(aCurr) if aCurr >= aMax => Some(aMax)
          case Some(aCurr) if math.abs(aHi - aLo) > EPS_MIN =>
            val phiACurr = phi(aCurr)
            if (phiACurr > phiZero + c1 * aCurr * dPhiZero || phiACurr >= phi(aLo)) {
              zoom(aLo, aCurr)
            } else {
              val dPhiCurr = dPhi(aCurr)
              if (math.abs(dPhiCurr) <= -c2 * dPhiZero) {
                Some(aCurr)
              } else if (dPhiCurr * (aHi - aLo) >= 0) {
                zoom(aCurr, aLo)
              } else {
                zoom(aCurr, aHi)
              }
            }
          case _ => Some((aHi + aLo) / 2D)
        }
      }

      /**
        * Finds the minimizer of the Cubic interpolation of the line search
        * objective \phi(\alpha) between [alpha_{i-1}, alpha_i]. See Nocedal (3.59).
        **/
      def interpolate(prev: Double, curr: Double): Option[Double] = {
        val d1 = dPhi(prev) + dPhi(curr) - 3D * (phi(prev) - phi(curr)) / (prev - curr)
        val d2 = signum(curr - prev) * sqrt(pow(d1, 2) - dPhi(prev) * dPhi(curr))
        val res = curr - (curr - prev) * (dPhi(curr) + d2 - d1) / (dPhi(curr) - dPhi(prev) + 2D * d2)

        if (!res.isNaN) Some(res) else None
      }
      findAlpha(0, phiZero, aMax*initialBracketRatio, firstIter = true)
  }

  /** Restricts a vector function `f` with derivative `df` along ray `f(x + alpha * p)` */
  private[optala] def restrictRay(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x: Vector[Double],
      p: Vector[Double]): (Double => Double, Double => Double) = {
    (alpha => f(x + alpha * p), alpha => df(x + alpha * p) dot p)
  }


  /**
    * Computes the exact step size alpha required to minimize a quadratic form.
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
