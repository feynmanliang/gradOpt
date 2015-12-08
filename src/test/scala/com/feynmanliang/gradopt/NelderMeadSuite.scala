package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._

import org.scalatest._

class NelderMeadSuite extends FunSpec {
  describe("Nedler-Mead") {
    val nm = new NelderMeadOptimizer(maxSteps = 5000, tol = 1E-8)

    describe("when applied to f(x,y) = x^2 + y^2") {
      val f: Vector[Double] => Double = x => x dot x
      val xOpt = DenseVector(0D, 0D)

      describe(s"when initialized using a known ``good'' simplex") {
        val tol = 1E-4
        var init = Simplex(
          List(
            DenseVector(-1D,.1D),
            DenseVector(-.1D,-3D),
            DenseVector(-2D,7D)).map(x => (x,f(x))))
          nm.minimize(f, init, true) match {
            case (Some(xStar), Some(perf)) => {
              val numIters = perf.xTrace.size
              it("should have at least one iteration") {
                assert(numIters >= 1)
              }
              it(s"should have evaluated f >= $numIters times") {
                assert(perf.numEvalF >= numIters)
              }
              it(s"should have not have evaluated df") {
                assert(perf.numEvalDf == 0)
              }
              it(s"should be within $tol to $xOpt") {
                assert(norm((xStar - xOpt).toDenseVector) < tol)
              }
              it(s"should monotonically decrease average function value (since obj is convex)") {
                val avgFVals = perf.xTrace.map(_._2)
                assert(avgFVals.sliding(2).forall(x => x(0) >= x(1)))
              }
            }
            case _ => fail("failed to return answer or perf diagnostics")
          }
        }

        describe(s"when the initial simplex is automatically initialized ") {
          nm.minimize(f, 2, 5, true) match {
            case (_, Some(perf)) => {
              it(s"should monotonically decrease average function value (since obj is convex)") {
                val avgFVals = perf.xTrace.map(_._2)
                assert(avgFVals.sliding(2).forall(x => x(0) >= x(1)))
              }
            }
            case _ => fail("failed to return perf diagnostics")
          }
        }
      }
    }
  }

  // vim: set ts=2 sw=2 et sts=2:
