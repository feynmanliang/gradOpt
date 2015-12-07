package com.feynmanliang.gradopt

import org.scalatest._

class OptimizerSuite extends FunSpec {
  val opt = new Optimizer()

  describe("Bracketing") {
    describe("when applied to f(x)=x^2") {
      val f = (x:Double) => x*x

      for (x0 <- List(-32D, 4D, 0D, 3D, 50D)) {
        describe(s"when initialized at x0=${x0}") {
          it("should return a convex interval ") {
            opt.bracket(f, x0) match {
              case Some(BracketInterval(lb, mid, ub)) => {
                assert(f(lb) > f(mid) && f(ub) > f(mid))
              }
              case _ => fail("No bracket interval returned!")
            }
          }
        }
      }
    }

    describe("when applied to f(x) = x") {
      val f = (x:Double) => x

      it ("should not find a bracket region") {
        assert(opt.bracket(f, 0).isEmpty)
      }
    }
  }

  describe("Line search") {
    describe("when applied to f(x) = x^2") {
      val f = (x:Double) => x*x

      for (x <- List(-17, 0, 4)) {
        val bracket = opt.bracket(f, x).get // safe, know x^2 is convex
        val xnew = opt.lineSearch(f, x, bracket)
        describe(s"when initialized with x=${x}") {
          it("should return a point within the bracket") {
            assert(bracket.contains(xnew))
          }
          it("should not increase f(x)") {
            assert(f(xnew) <= f(x))
          }
        }
      }
    }
  }

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

      val f = (x:Double) => x*x
      val df = (x:Double) => 2.0*x

      for (x0 <- List(-17.3, 0.1, 4.2)) {
        describe(s"when initialized at x0=$x0") {
          opt.minimize(f, df, x0, reportPerf = true) match {
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
                assert(math.abs(xstar - xopt) < tol)
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
