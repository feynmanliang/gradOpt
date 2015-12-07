package com.feynmanliang.gradopt

import org.scalatest._

class OptimizerSuite extends FunSuite {
  test("Bracketing should return an interval where f(x)=x^2 is convex") {
    val f = (x:Double) => x*x
    val df = (x:Double) => 2D * x

    val opt = new Optimizer()

    opt.bracket(f, 0D) match {
      case Some(BracketInterval(lb, mid, ub)) => {
        assert(f(lb) > f(mid) && f(ub) > f(mid))
      }
      case _ => fail("No bracket interval returned!")
    }

    opt.bracket(f, 8D) match {
      case Some(BracketInterval(lb, mid, ub)) => {
        assert(f(lb) > f(mid) && f(ub) > f(mid))
      }
      case _ => fail("No bracket interval returned!")
    }
  }
}

// vim: set ts=2 sw=2 et sts=2:
