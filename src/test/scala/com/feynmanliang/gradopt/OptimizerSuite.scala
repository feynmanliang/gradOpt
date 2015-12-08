package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._

import org.scalatest._

import com.feynmanliang.gradopt.GradientAlgorithm._
import com.feynmanliang.gradopt.LineSearchConfig._

class OptimizerSuite extends FunSpec {
  val tol = 1E-5 // tolerance for norm(x* - xOpt)
  val opt = new Optimizer(maxSteps=30000, tol=1E-6)

  describe("FunctionWithCounter") {
    it("correctly counts the number of times a function is called") {
      for (n <- 10 to 1000 by 100) {
        val cntFn = new FunctionWithCounter((x: Double) => x * x)
        for (i <- 0 until n) {
          cntFn(i)
        }
        assert(cntFn.numCalls == n)
      }
    }
  }

  describe("Minimize") {

    for {
      gradientAlgorithm <- List(SteepestDescent, ConjugateGradient)
    } describe (s"using $gradientAlgorithm") {
      // Convex => unique minima at xOpt
      case class ConvexTestCase(
        name: String,
        f: Vector[Double] => Double,
        df: Vector[Double] => Vector[Double],
        xOpt: Vector[Double],
        xInits: List[Vector[Double]])

      val testCases = List(
        ConvexTestCase(
          "f(x) = x^2",
          v => v dot v,
          x => 2D*x,
          Vector(0D),
          List(-17.3, 0.1, 4.2).map(DenseVector(_))),
        ConvexTestCase(
          "f(x,y) = (x-1)^2 + (y-2)^2",
          v => pow(norm(v.toDenseVector - DenseVector(1D,2D)), 2),
          x => 2D*(x.toDenseVector - DenseVector(1D,2D)),
          DenseVector(1D,2D),
          List(DenseVector(-17.3,2), DenseVector(0.1,-4), DenseVector(3,4.2))),
        ConvexTestCase(
          "f(x,y) = (1 - x)^2 + 100 (y - x^2)^2",
          v => pow(1D - v(0), 2) + 100D * pow(v(1) - pow(v(0), 2),2),
          v => {
            DenseVector(
              -2D*(1 - v(0)) - 400D * v(0) * (-pow(v(0), 2) + v(1)),
              200D * (-pow(v(0), 2) + v(1))
            )
          },
          DenseVector(1D,1D),
          List(
            DenseVector(-3D,-4D)
          )
        )
      )

      for {
        ConvexTestCase(name, f, df, xOpt, xInits) <- testCases
      } describe(s"when applied to $name") {
        for {
          x0 <- xInits
        } describe(s"when initialized at $x0") {
          opt.minimize(f, df, x0, gradientAlgorithm, CubicInterpolation, true) match {
            case (Some(xStar), Some(perf)) => {
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
              it(s"should be within $tol to $xOpt") {
                assert(norm((xStar - xOpt).toDenseVector) < tol)
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
