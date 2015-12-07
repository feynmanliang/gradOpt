package com.feynmanliang.gradopt

import org.scalatest._

class OptimizerSuite extends FunSuite {
  test("Bracketing should return an interval where f(x)=x^2 is convex") {
    def f(x:Double): Double = x*x
    def g(x:Double): Double = 2D * x
    val opt = new Optimizer()
    val bracket = opt.bracket(f _, g _, 0)
    bracket match {
      case BracketInterval(lb, mid, ub) =>
        assert(f(lb) > f(mid) && f(ub) > f(mid))
    }
  }
}

// vim: set ts=2 sw=2 et sts=2:
