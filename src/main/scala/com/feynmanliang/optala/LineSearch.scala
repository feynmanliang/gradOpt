package com.feynmanliang.optala

import breeze.linalg._
import breeze.numerics._

object LineSearch {
  val aMax = 100D
  val tol = 1E-16

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
      c2: Double = 0.9): Option[Double] = {
    val (phi, dPhi) = restrictRay(f, df, x, p)
    val phiZero = phi(0)
    val dPhiZero = dPhi(0)

    /**
      * Nocedal Algorithm 3.5, finds a step length alpha while ensures that
      * (aPrev, aCurr) contains a point satisfying the Strong Wolfe Conditions at
      * each iteration.
      */
    def bracket(aPrev: Double, phiPrev: Double, aCurr: Double, firstIter: Boolean): Option[Double] = {
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
          bracket(aCurr, phiCurr, (aCurr + aMax) / 2D, firstIter=false)
        }
      }
    }

    /**
      * Nocedal Algorithm 3.6, generates \alpha_j between \alpha_{lo} and \alpha_{hi} and replaces
      * one of the two endpoints while ensuring Wolfe conditions hold.
      */
    def zoom(aLo: Double, aHi: Double): Option[Double] = {
      assert(!aLo.isNaN && !aHi.isNaN)
      interpolate(aLo, aHi) match {
        case Some(aCurr) if math.abs(aHi - aLo) > tol =>
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

    bracket(0, phiZero, aMax * 1E-6, firstIter = true)
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
