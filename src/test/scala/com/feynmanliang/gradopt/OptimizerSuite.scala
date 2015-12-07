package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._

import org.scalatest._


class OptimizerSuite extends FunSpec {
  val opt = new Optimizer()

  describe("FunctionWithCounter") {
    it("correctly counts the number of times a function is called") {
      for (n <- 10 to 1000 by 100) {
        val opt = new Optimizer()
        val cntFn = new opt.FunctionWithCounter((x: Double) => x * x)
        for (i <- 0 until n) {
          cntFn(i)
        }
        assert(cntFn.numCalls == n)
      }
    }
  }

  describe("Minimize") {
    describe("when applied to f(x) = x^2") {
      val tol = 1E-6
      val xopt = 0.0D

      val f: Vector[Double] => Double = v => v dot v
      val df: Vector[Double] => Vector[Double] = x => {
        require(x.length == 1, "df only defined R -> R")
        2D*x
      }

      for (x0 <- List(-17.3, 0.1, 4.2).map(DenseVector(_))) {
        describe(s"when initialized at x0=$x0") {
          opt.minimize(f, df, x0, true) match {
            case (Some(xstar), Some(perf)) => {
              val numIters = perf.xTrace.size
              it("should have at least one iteration") {
                assert(numIters >= 1)
              }
              it(s"should have evaluated f >= $numIters times") {
                assert(perf.numEvalF > numIters)
              }
              it(s"should have evaluated df >= $numIters times") {
                assert(perf.numEvalDf > numIters)
              }
              it(s"should be within $tol to $xopt") {
                assert(norm((xstar - xopt).toDenseVector) < tol)
              }
            }
            case _ => fail("Minimize failed to return answer or perf diagnostics")
          }
        }
      }
    }

    describe("when applied to f(x,y) = (x-1)^2 + (y-2)^2") {
      val tol = 1E-6
      val xopt = DenseVector(1D,2D)

      val f: Vector[Double] => Double = v => {
        val tmp = v.toDenseVector - DenseVector(1D,2D)
        tmp dot tmp
      }
      val df: Vector[Double] => Vector[Double] = x => {
        require(x.length == 2, "df only defined R^2 -> R^2")
        2D*(x.toDenseVector - DenseVector(1D,2D))
      }
      for (x0 <- List(
        DenseVector(-17.3,2),
        DenseVector(0.1,-4),
        DenseVector(3,4.2)
      )) {
        describe(s"when initialized at x0=$x0") {
          opt.minimize(f, df, x0, true) match {
            case (Some(xstar), Some(perf)) => {
              val numIters = perf.xTrace.size
              it("should have at least one iteration") {
                assert(numIters >= 1)
              }
              it(s"should have evaluated f >= $numIters times") {
                assert(perf.numEvalF > numIters)
              }
              it(s"should have evaluated df >= $numIters times") {
                assert(perf.numEvalDf > numIters)
              }
              it(s"should be within $tol to $xopt") {
                assert(norm((xstar - xopt).toDenseVector) < tol)
              }
            }
            case _ => fail("Minimize failed to return answer or perf diagnostics")
          }
        }
      }
    }
  }
}

// vim: set ts=2 sw=2 et sts=2:
