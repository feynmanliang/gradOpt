package com.feynmanliang.optala

import breeze.linalg._
import org.scalatest._

class LineSearchSuite extends FunSpec {
  describe("Approximate line search") {
    describe("when applied to f(x) = x^2") {
      val f: Vector[Double] => Double = v => v dot v
      val df: Vector[Double] => Vector[Double] = x => {
        require(x.length == 1, "df only defined R -> R")
        2D*x
      }

      for (x <- List(-17D, 0D, 4D).map(DenseVector(_))) {
        val xnew = x - LineSearch.chooseStepSize(f, df, x, -df(x)).get * df(x)
        describe(s"when initialized with x=$x") {
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
