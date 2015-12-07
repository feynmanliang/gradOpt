package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._

// A bracketing interval where f(mid) < f(lb) and f(mid) < f(ub), guaranteeing a minimum
case class BracketInterval(lb: Double, mid: Double, ub: Double)

class Optimizer {

  /**
   * Brackets the minimum of a scalar function `f` with gradient `g`. This function uses `x0`
   * as the midpoint around which to identify the bracket bounds.
   */
  def bracket(f: Double => Double, df: Double => Double, x0: Double): BracketInterval = {
    ???
  }
}

// vim: set ts=2 sw=2 et sts=2:
