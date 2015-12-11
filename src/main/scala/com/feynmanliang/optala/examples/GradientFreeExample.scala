package com.feynmanliang.optala.examples

import breeze.linalg.{DenseVector, Vector}
import breeze.numerics.pow

import com.feynmanliang.optala.{GeneticAlgorithm, NelderMeadOptimizer}

object GradientFreeExample {
  def main(args: Array[String]) {
    val f: Vector[Double] => Double = v => {
      val x = v(0)
      val y = v(1)
      (4D - 2.1D * pow(x, 2) + (1D / 3D) * pow(x, 4)) * pow(x, 2) + x * y + (4D * pow(y, 2) - 4D) * pow(y, 2)
    }
    val lb = DenseVector(-2D, -1D)
    val ub = DenseVector(2D, 1D)


    println(s"Optimizing 6HCF using Nedler-Mead")
    val nmOpt = new NelderMeadOptimizer(maxSteps = 1000, tol = 1E-10)
    val x0 = DenseVector(-1D, -1D)
    nmOpt.minimize(f, 2, 8, reportPerf = true) match {
      case (_, Some(perf)) =>
        val (sstar, fstar) = perf.stateTrace.last
        val xstar = sstar.points.map(_._1).reduce(_ + _) / (1D * sstar.points.size)
        println(s"x0:$x0,xstar:$xstar,fstar:$fstar,normGrad:${perf.stateTrace.last._2}," +
          s"numSteps:${perf.stateTrace.length},fEval:${perf.numObjEval},dfEval:${perf.numGradEval}")
      case _ => println(s"No results for x0=$x0!!!")
    }

    println(s"Optimizing 6HCF using GA")
    val ga = new GeneticAlgorithm(maxSteps = 1000)
    val popSize = 20
    val eliteCount = 2
    val xoverFrac = 0.8
    val seed = 42
    ga.minimize(f, lb, ub, popSize, eliteCount, xoverFrac, Some(seed)) match {
      case (_, Some(perf)) =>
        val (xstar, fstar) = perf.stateTrace.last._1.population.minBy(_._2)
        println(s"popSize:$popSize,xstar:$xstar,fstar:$fstar,numSteps:${perf.stateTrace.length}," +
          s"fEval:${perf.numObjEval},dfEval:${perf.numGradEval}")
      case _ => println(s"No results for x0=$x0!!!")
    }
  }
}
