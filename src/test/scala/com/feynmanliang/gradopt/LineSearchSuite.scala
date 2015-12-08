package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._

import org.scalatest._


class LineSearchSuite extends FunSpec {
  val opt = new Optimizer()

  describe("Line search") {
    describe("when applied to f(x) = x^2") {
      val f: Vector[Double] => Double = v => v dot v
      val df: Vector[Double] => Vector[Double] = x => {
        require(x.length == 1, "df only defined R -> R")
        2D*x
      }

      for (x <- List(-17D, 0D, 4D).map(DenseVector(_))) {
        val bracket = opt.bracket(f, df, x).get // safe, know x^2 is convex
        val xnew = x - opt.lineSearch(f, df, x, bracket) * df(x)
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
}
