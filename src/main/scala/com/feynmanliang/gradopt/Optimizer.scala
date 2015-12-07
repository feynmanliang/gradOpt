package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._

// A bracketing interval where f(mid) < f(lb) and f(mid) < f(ub), guaranteeing a minimum
case class BracketInterval(lb: Double, mid: Double, ub: Double) {
  def contains(x: Double): Boolean = lb <= x && ub >= x
  def size: Double = ub - lb
}

class Optimizer {

  /**
   * Brackets the minimum of a scalar function `f`. This function uses `x0` as
   * the midpoint around which to identify the bracket bounds.
   */
  def bracket(f: Double => Double, x0: Double): Option[BracketInterval] = {
    def doublingBrackets(currBracket: BracketInterval): Stream[BracketInterval] =
      currBracket match {
        case BracketInterval(lb, mid, ub) => {
          val delta = mid - lb
          val newLb = if (f(mid) < f(lb)) lb else lb - delta
          val newUb = if (f(mid) < f(ub)) ub else ub + delta
          currBracket #:: doublingBrackets(BracketInterval(newLb, mid, newUb))
        }
      }

    val initBracket = BracketInterval(x0 - 1D, x0, x0 + 1D)

    doublingBrackets(initBracket)
      .take(5000) // stops after 5000 brackets, TODO: more robust stopping criterion
      .find(_ match {
          case BracketInterval(lb, mid, ub) => f(lb) > f(mid) && f(ub) > f(mid)
      })
  }

  /**
   * Performs a line search for x' = x + a*p within a bracketing interval to determine step size.
   * This method linearly interpolates the bracket interval and chooses the minimizer of f.
   * TODO: bisection search the candidates
   */
   def lineSearch(f: Double => Double, x: Double, bracket: BracketInterval): Double = {
     val numPoints = 100D // number of points to interpolate within bracket interval
     val candidates = x +: (bracket.lb to bracket.ub by bracket.size/numPoints)
     candidates.minBy(f)
   }
}

// vim: set ts=2 sw=2 et sts=2:
