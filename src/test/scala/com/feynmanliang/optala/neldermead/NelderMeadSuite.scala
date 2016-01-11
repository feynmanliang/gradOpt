package com.feynmanliang.optala.neldermead

import breeze.linalg._
import com.feynmanliang.optala.Solution
import org.scalatest._

class NelderMeadSuite extends FunSpec {
  describe("Nedler-Mead") {
    val maxObjEvals = 1000

    describe("when applied to f(x,y) = x^2 + y^2") {
      val f: Vector[Double] => Double = x => x dot x
      val xOpt = DenseVector(0D, 0D)

      describe(s"when initialized using a known ``good'' simplex and terminated due to convergence") {
        val nm = new NelderMeadOptimizer(
          maxObjEvals = Int.MaxValue,
          maxIter = Int.MaxValue,
          tol = 1E-8)
        val tol = 1E-4
        val init = Simplex(List(
          DenseVector(-1D, .1D),
          DenseVector(-.1D, -3D),
          DenseVector(-2D, 7D)).map(x => Solution(f, x)))
        val result = nm.minimize(f, init)
        val numIters = result.stateTrace.size
        it("should have at least one iteration") {
          assert(numIters >= 1)
        }
        it(s"should have evaluated f >= $numIters times") {
          assert(result.numObjEval >= numIters)
        }
        it(s"should have not have evaluated df") {
          assert(result.numGradEval == 0)
        }
        it(s"should be within $tol to $xOpt") {
          val xStar = result.stateTrace.last.bestSolution.point
          assert(norm((xStar - xOpt).toDenseVector) < tol)
        }
        it(s"should monotonically decrease average function value (since obj is convex)") {
          assert(result.stateTrace.sliding(2).forall(x => x(1).averageObjVal <= x.head.averageObjVal))
        }

        describe(s"when the initial simplex is automatically initialized and terminated on numObjectiveEvals") {
          val nm = new NelderMeadOptimizer(
            maxObjEvals = maxObjEvals,
            maxIter = Int.MaxValue,
            tol = 1E-8)
          val result = nm.minimize(f, 2, 5)
          it(s"should monotonically decrease average function value (since obj is convex)") {
            assert(result.stateTrace.sliding(2).forall(x => x(1).averageObjVal <= x.head.averageObjVal))
          }
          it(s"terminates after evaluating objective function $maxObjEvals times") {
            // assumes each iteration evaluates objective less that 0.2*maxObjectiveEvals
            assert(result.numObjEval <= (maxObjEvals * 1.2).toInt)
          }
        }
      }
    }
  }
}
