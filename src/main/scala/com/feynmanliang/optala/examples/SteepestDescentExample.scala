package com.feynmanliang.optala.examples

import breeze.linalg.linspace
import breeze.numerics.{sin, cos, pow}
import breeze.plot.plot
import breeze.plot.Figure

import com.feynmanliang.optala.GradientOptimizer
import com.feynmanliang.optala.GradientAlgorithm._
import com.feynmanliang.optala.LineSearchConfig._

object SteepestDescentExample {
  def main(args: Array[String]) {
    val showPlot = args(0).toBoolean

    val f = (x: Double) => pow(x, 4) * cos(pow(x, -1)) + 2D * pow(x, 4)
    val df = (x: Double) => 8D * pow(x, 3) + 4D * pow(x, 3) * cos(pow(x, -1)) - pow(x, 4) * sin(pow(x, -1))

    if (showPlot) {
      val fig = Figure()
      val x = linspace(-.1, 0.1)
      fig.subplot(2, 1, 0) += plot(x, x.map(f))
      fig.subplot(2, 1, 1) += plot(x, x.map(df))
      fig.saveas("lines.png") // save current figure as a .png, eps and pdf also supported
    }

    val gradOpt = new GradientOptimizer(maxSteps = 5000, tol = 1E-6)
    for (x0 <- List[Double](-5, -1, -0.1, -1E-2, -1E-3, -1E-4, -1E-5, 1E-5, 1E-4, 1E-3, 1E-2, 0.1, 1, 5)) {
      gradOpt.minimize(f, df, x0, SteepestDescent, CubicInterpolation, reportPerf = true) match {
        case (_, Some(perf)) =>
          val (xstar, fstar) = perf.stateTrace.last
          println(s"x0:$x0,xstar:$xstar,fstar:$fstar,normGrad:${perf.stateTrace.last._2}," +
            s"numSteps:${perf.stateTrace.length},fEval:${perf.numObjEval},dfEval:${perf.numGradEval}")
        case _ => println(s"No results for x0=$x0!!!")
      }
    }
  }
}
