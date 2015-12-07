package com.feynmanliang.gradopt

import org.scalatest._

class OptimizerSuite extends FunSpec {
  describe("Bracketing") {
    describe("when applied to f(x)=x^2") {
      val f = (x:Double) => x*x
      val opt = new Optimizer()

      it("should return a convex interval when initialized at [-32, -4, 0, 3, 50]") {
        for (i <- List(-32D, 4D, 0D, 3D, 50D)) {
          opt.bracket(f, i) match {
            case Some(BracketInterval(lb, mid, ub)) => {
              assert(f(lb) > f(mid) && f(ub) > f(mid))
            }
            case _ => fail("No bracket interval returned!")
          }
        }
      }
    }

    describe("when applied to f(x) = x") {
      val f = (x:Double) => x
      val opt = new Optimizer()

      it ("should not find a bracket region") {
        assert(opt.bracket(f, 0).isEmpty)
      }
    }
  }
}

// vim: set ts=2 sw=2 et sts=2:
