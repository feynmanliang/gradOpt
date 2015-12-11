package com.feynmanliang.optala.examples

import java.io.File

import breeze.linalg.{DenseVector, DenseMatrix, csvread}

import com.feynmanliang.optala.GradientOptimizer
import com.feynmanliang.optala.GradientAlgorithm._
import com.feynmanliang.optala.LineSearchConfig._

object QuadraticFormExample {
  def main(args: Array[String]) {
    val gradOpt = new GradientOptimizer(maxSteps = 101, tol = 1E-4)
    for {
      lsAlgo <- List(Exact, CubicInterpolation)
      fname <- List("A10.csv", "A100.csv", "A1000.csv", "B10.csv", "B100.csv", "B1000.csv")
      optAlgo <- List(SteepestDescent, ConjugateGradient)
    } {
      val A: DenseMatrix[Double] = csvread(new File(getClass.getResource("/" + fname).getFile))
      assert(A.rows == A.cols, "A must be symmetric")
      val n: Int = A.cols
      val b: DenseVector[Double] = 2D * (DenseVector.rand(n) - DenseVector.fill(n) {
        0.5
      })

      println(s"lsAlgo:$lsAlgo,fname:$fname,optAlgo:$optAlgo")
      gradOpt.minQuadraticForm(A, b, DenseVector.zeros(n), optAlgo, lsAlgo, reportPerf = true) match {
        case (res, Some(perf)) =>
          println(s"normGrad:${perf.stateTrace.last._2},numSteps:${perf.stateTrace.length}," +
            s"fEval:${perf.numObjEval},dfEval:${perf.numGradEval}")
        case _ => throw new Exception("Minimize failed to return perf diagnostics")
      }
    }
  }

}
