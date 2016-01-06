package com.feynmanliang.optala.examples

import breeze.linalg.DenseMatrix
import breeze.numerics.{cos, pow, sin}
import com.feynmanliang.optala.GradientAlgorithm._
import com.feynmanliang.optala.GradientOptimizer
import com.feynmanliang.optala.LineSearchConfig._

object SteepestDescentExample {
  def main(args: Array[String]) {
    val f = (x: Double) => pow(x, 4) * cos(pow(x, -1)) + 2D * pow(x, 4)
    val df = (x: Double) => 8D * pow(x, 3) + 4D * pow(x, 3) * cos(pow(x, -1)) - pow(x, 4) * sin(pow(x, -1))

    val gradOpt = new GradientOptimizer(maxSteps = 10000, tol = 1E-6)
    for {
      x0 <- List[Double](-50, -10, -5, -1, -0.1, -1E-2, -1E-3, -1E-4, -1E-5, 1E-5, 1E-4, 1E-3, 1E-2, 0.1, 1, 5, 10, 50)
    } {
      gradOpt.minimize(f, df, x0, SteepestDescent, CubicInterpolation, reportPerf = true) match {
        case (_, Some(perf)) =>
          println(f"$x0 & ${perf.stateTrace.size} & ${perf.numObjEval} & ${perf.numGradEval}" +
            f" & ${perf.bestSolution.objVal}%.3E & ${perf.bestSolution.normGrad}%.3E\\\\")
          DenseMatrix(x0 +: perf.stateTrace.flatMap(_.point.toArray):_*)
        case _ => sys.error(s"No results for x0=$x0!!!")
      }
    }
  }
}
