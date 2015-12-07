package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._

import org.scalatest._


class BracketingSuite extends FunSpec {
  val opt = new Optimizer()

  describe("Bracketing") {
    describe("when applied to f(x)=x^2") {
      val f: Vector[Double] => Double = v => v dot v
      val df: Vector[Double] => Vector[Double] = x => {
        require(x.length == 1, "df only defined R -> R")
        2D*x
      }

      for (x0 <- List(-32D, -4D, 0D, 3D, 50D).map(DenseVector(_))) {
        describe(s"when initialized at x0=${x0}") {
          it("should return a convex interval ") {
            opt.bracket(f, df, x0) match {
              case Some(BracketInterval(lb, mid, ub)) => {
                val dfx0 = df(x0)
                assert(f(x0 - lb*dfx0) >= f(x0 - mid*dfx0) && f(x0 - ub*dfx0) >= f(x0 - mid*dfx0))
              }
              case _ => fail("No bracket interval returned!")
            }
          }
        }
      }
    }

    describe("when applied to f(x) = x") {
      val f = (x:Vector[Double]) => x(0)
      val df = (x:Vector[Double]) => DenseVector(1D)

      it("should not find a bracket region") {
        assert(opt.bracket(f, df, DenseVector(0)).isEmpty)
      }
    }
  }
}
