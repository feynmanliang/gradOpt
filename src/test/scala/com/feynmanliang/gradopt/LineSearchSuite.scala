package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._

import org.scalatest._


class LineSearchSuite extends FunSpec {
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
            LineSearch.bracket(f, df, x0) match {
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
        assert(LineSearch.bracket(f, df, DenseVector(0)).isEmpty)
      }
    }
  }

  describe("Approximate line search") {
    describe("when applied to f(x) = x^2") {
      val f: Vector[Double] => Double = v => v dot v
      val df: Vector[Double] => Vector[Double] = x => {
        require(x.length == 1, "df only defined R -> R")
        2D*x
      }

      for (x <- List(-17D, 0D, 4D).map(DenseVector(_))) {
        val bracket = LineSearch.bracket(f, df, x).get // safe, know x^2 is convex
        val xnew = x - LineSearch.chooseStepSize(f, -df(x), df, x).get * df(x)
        describe(s"when initialized with x=${x}") {
          it("should return a point within the bracket") {
            val p = if (norm(df(x).toDenseVector) == 0D) 0D else norm((x - xnew).toDenseVector) / norm(df(x).toDenseVector)
            assert(bracket.contains(p))
          }
          it("should not increase f(x)") {
            assert(f(xnew) <= f(x))
          }
        }
      }
    }
  }

  describe("Exact line search") {
    describe("When applied to A=[1 0; 0 1], b=[-2 -3]") {
      val A = DenseMatrix((1D, 0D), (0D, 1D))
      val b = DenseVector(-2D, -3D)
      val x = DenseVector(2D, 4D)
      it("should converge in a single step") {
        val p = -(A*x - b) // steepest descent direction
        val xNew = x + LineSearch.exactLineSearch(A, b, p, x) * p
        assert(norm(A*xNew - b) <= 1E-6)
      }
    }
  }
}
