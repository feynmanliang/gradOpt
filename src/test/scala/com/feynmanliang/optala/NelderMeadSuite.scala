package com.feynmanliang.optala

import breeze.linalg._
import org.scalatest._

class NelderMeadSuite extends FunSpec {
  describe("Nedler-Mead") {
    val maxObjEvals = 1000

    describe("when applied to f(x,y) = x^2 + y^2") {
      val f: Vector[Double] => Double = x => x dot x
      val xOpt = DenseVector(0D, 0D)

      describe(s"when initialized using a known ``good'' simplex and terminated due to convergence") {
        val nm = new NelderMeadOptimizer(
          maxObjectiveEvals = Int.MaxValue,
          maxSteps = Int.MaxValue,
          tol = 1E-8)
        val tol = 1E-4
        val init = Simplex(
          List(
            DenseVector(-1D,.1D),
            DenseVector(-.1D,-3D),
            DenseVector(-2D,7D)).map(x => (x,f(x))))
          nm.minimize(f, init, reportPerf = true) match {
            case (Some(xStar), Some(perf)) =>
              val numIters = perf.stateTrace.size
              it("should have at least one iteration") {
                assert(numIters >= 1)
              }
              it(s"should have evaluated f >= $numIters times") {
                assert(perf.numObjEval >= numIters)
              }
              it(s"should have not have evaluated df") {
                assert(perf.numGradEval == 0)
              }
              it(s"should be within $tol to $xOpt") {
                assert(norm((xStar - xOpt).toDenseVector) < tol)
              }
              it(s"should monotonically decrease average function value (since obj is convex)") {
                val avgFVals = perf.stateTrace.map( _.points.map(_._2).sum)
                assert(avgFVals.sliding(2).forall(x => x.head >= x(1)))
              }
            case _ => fail("failed to return answer or perf diagnostics")
          }
        }

        describe(s"when the initial simplex is automatically initialized and terminated on numObjectiveEvals") {
          val nm = new NelderMeadOptimizer(
            maxObjectiveEvals = maxObjEvals,
            maxSteps = Int.MaxValue,
            tol = 1E-8)
          nm.minimize(f, 2, 5, reportPerf = true) match {
            case (_, Some(perf)) =>
              it(s"should monotonically decrease average function value (since obj is convex)") {
                val avgFVals = perf.stateTrace.map(_.points.map(_._2).sum)
                assert(avgFVals.sliding(2).forall(x => x.head >= x(1)))
              }
              it(s"terminates after evaluating objective function $maxObjEvals times") {
                // assumes each iteration evaluates objective less that 0.2*maxObjectiveEvals
                assert(perf.numObjEval <= (maxObjEvals*1.2).toInt)
              }
            case _ => fail("failed to return perf diagnostics")
          }

        }
      }
    }
  }
