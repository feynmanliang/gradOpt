package com.feynmanliang.optala.examples

import java.io.File

import breeze.linalg.{DenseVector, DenseMatrix, csvread}

import com.feynmanliang.optala.GradientOptimizer
import com.feynmanliang.optala.GradientAlgorithm._
import com.feynmanliang.optala.LineSearchConfig._

object QuadraticFormExample {
  def main(args: Array[String]) {
    val gradOpt = new GradientOptimizer(maxSteps = 100, tol = 1E-6)
    for {
      optAlgo <- List(SteepestDescent, ConjugateGradient)
    } yield {
      println(s"====$optAlgo=====")
      for {
        fname <- List("A10", "A100", "A1000", "B10", "B100", "B1000")
        lsAlgo <- List(Exact, CubicInterpolation)
      } yield {
        val A: DenseMatrix[Double] = csvread(new File(getClass.getResource("/" + fname + ".csv").getFile))
        assert(A.rows == A.cols, "A must be symmetric")
        val n: Int = A.cols
        val b: DenseVector[Double] = 2D * (DenseVector.rand(n) - DenseVector.fill(n) {
          0.5
        })

        gradOpt.minQuadraticForm(A, b, DenseVector.zeros(n), optAlgo, lsAlgo, reportPerf = true) match {
          case (res, Some(perf)) =>
            println(s"$lsAlgo & $fname & ${perf.stateTrace.size} & ${perf.numObjEval} & ${perf.numGradEval}" +
              f" & ${perf.stateTrace.last._1(0)}%.3E & ${perf.stateTrace.last._2}%.3E\\\\")
          case _ => throw new Exception("Minimize failed to return perf diagnostics")
        }
      }
    }
  }
}
