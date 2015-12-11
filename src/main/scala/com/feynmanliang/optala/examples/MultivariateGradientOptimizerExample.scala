package com.feynmanliang.optala.examples

import breeze.linalg.{Vector, DenseVector}
import breeze.numerics.pow

import com.feynmanliang.optala.{GradientOptimizer, NelderMeadOptimizer}
import com.feynmanliang.optala.GradientAlgorithm._
import com.feynmanliang.optala.LineSearchConfig._

object MultivariateGradientOptimizerExample {
  def main(args: Array[String]) {
    val f: Vector[Double] => Double = v => pow(1D - v(0), 2) + 100D * pow(v(1) - pow(v(0), 2), 2)
    val df: Vector[Double] => Vector[Double] = v => {
      DenseVector(
        -2D * (1 - v(0)) - 400D * v(0) * (-pow(v(0), 2) + v(1)),
        200D * (-pow(v(0), 2) + v(1))
      )
    }
    val xOpt = (-1D, 1D)
    val x0 = DenseVector(-3D, -4D)

    val gradOpt = new GradientOptimizer(maxSteps = 5000, tol = 1E-6)
    for (algo <- List(SteepestDescent, ConjugateGradient)) {
      println(s"Optimizing Rosenbrock function using $algo")
      gradOpt.minimize(f, df, x0, algo, CubicInterpolation, reportPerf = true) match {
        case (_, Some(perf)) =>
          val (xstar, fstar) = perf.stateTrace.last
          println(s"x0:$x0,xstar:$xstar,fstar:$fstar,normGrad:${perf.stateTrace.last._2}," +
            s"numSteps:${perf.stateTrace.length},fEval:${perf.numObjEval},dfEval:${perf.numGradEval}")
        case _ => println(s"No results for x0=$x0!!!")
      }
    }

    val nmOpt = new NelderMeadOptimizer(maxSteps = 5000, tol = 1E-10)
    println(s"Optimizing Rosenbrock function using Nedler-Mead")
    nmOpt.minimize(f, 2, 8, reportPerf = true) match {
      case (_, Some(perf)) =>
        val sstar = perf.stateTrace.last
        val xstar = sstar.points.minBy(_._2)._1
        val fstar = f(xstar)
        println(s"x0:$x0,xstar:$xstar,fstar:$fstar," +
          s"numSteps:${perf.stateTrace.length},fEval:${perf.numObjEval},dfEval:${perf.numGradEval}")
      case _ => println(s"No results for x0=$x0!!!")
    }
  }
}
