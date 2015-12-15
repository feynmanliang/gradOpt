package com.feynmanliang.optala

import breeze.linalg._
import org.scalatest._

class LineSearchSuite extends FunSpec {
  describe("Bracketing") {
    describe("when applied to f(x)=x^2") {
      val f: Vector[Double] => Double = v => v dot v
      val df: Vector[Double] => Vector[Double] = x => {
        require(x.length == 1, "df only defined R -> R")
        2D*x
      }

      for (x0 <- List(-32D, -4D, 1E-3D, 3D, 50D).map(DenseVector(_))) {
        describe(s"when initialized at x0=$x0") {
          it("should return a convex interval ") {
            val dfx0 = df(x0)
            val p = -dfx0 / norm(dfx0.toDenseVector)
            LineSearch.bracket(f, df, x0, p) match {
              case Some(b) => assert(b.bracketsMin)
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
        assert(LineSearch.bracket(f, df, DenseVector(0), df(DenseVector(0))).isEmpty)
      }
    }
  }
  describe("Iterative line search") {
    describe("when applied to f(x) = x^2") {
      val f: Vector[Double] => Double = v => v dot v
      val df: Vector[Double] => Vector[Double] = x => {
        require(x.length == 1, "df only defined R -> R")
        2D*x
      }

      for (x <- List(-17D, 0D, 4D).map(DenseVector(_))) {
        val bracket = LineSearch.bracket(f, df, x, -df(x)).get // safe, know x^2 is convex
        val xnew: Vector[Double] = x - LineSearch.chooseStepSize(f, df, x, -df(x)).get * df(x)
        describe(s"when initialized with x=$x") {
          it("should return a point within the bracket") {
            val p = if (norm(df(x).toDenseVector) == 0D) 0D else norm(x - xnew) / norm(df(x).toDenseVector)
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
        val grad = A*x - b // steepest descent direction
        val p = (-1D/norm(grad.toDenseVector)) * grad
        val xNew = x + LineSearch.exactLineSearch(A, grad, x, p).get * p
        assert(norm(A*xNew - b) <= 1E-6)
      }
    }
  }
}
