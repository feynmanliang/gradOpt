package com.feynmanliang.optala.examples

import breeze.linalg.{DenseVector, Vector}
import breeze.numerics.pow
import breeze.stats.distributions.{Uniform, RandBasis, ThreadLocalRandomGenerator}

import com.feynmanliang.optala.{Simplex, GeneticAlgorithm, NelderMeadOptimizer}
import org.apache.commons.math3.random.MersenneTwister

object GradientFreeExample {
  def main(args: Array[String]) {
    val seed = 42L

    // This is the 6 Hump Camel Function (6HCF)
    val f: Vector[Double] => Double = v => {
      val x = v(0)
      val y = v(1)
      (4D - 2.1D * pow(x, 2) + (1D / 3D) * pow(x, 4)) * pow(x, 2) + x * y + (4D * pow(y, 2) - 4D) * pow(y, 2)
    }
    // Optimized over the region -2 <= x <= 2, -1 <= y <= 1
    val lb = DenseVector(-2D, -1D)
    val ub = DenseVector(2D, 1D)

    implicit val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

    println(s"Optimizing 6HCF using Nedler-Mead")
    val nmOpt = new NelderMeadOptimizer(maxSteps = 1000, tol = 1E-10)
    val initialSimplex = Simplex(Seq.fill(8){
      val simplexPoint = DenseVector(Uniform(-2D,2D).sample(), Uniform(-1D,1D).sample())
      (simplexPoint, f(simplexPoint))
    })
    nmOpt.minimize(f, initialSimplex, reportPerf = true) match {
      case (_, Some(perf)) =>
        val sstar = perf.stateTrace.last
        val xstar = sstar.points.minBy(_._2)._1
        val fstar = f(xstar)
        println(s"$xstar,\n" +
          s"fOpt:$fstar," +
          s"numIters:${perf.stateTrace.length}\n" +
          s"numObjEval: ${perf.numObjEval}\n" +
          s"numGradEval:${perf.numGradEval}")
      case _ => sys.error("No result found!")
    }

    println(s"Optimizing 6HCF using GA")
    val ga = new GeneticAlgorithm(maxSteps = 1000)
    val popSize = 20
    val eliteCount = 2
    val xoverFrac = 0.8
    ga.minimize(f, lb, ub, popSize, eliteCount, xoverFrac, Some(seed)) match {
      case (_, Some(perf)) =>
        val xstar = perf.stateTrace.last.population.minBy(_._2)._1
        val fstar = f(xstar)
        println(s"popSize:$popSize,xstar:$xstar,fstar:$fstar,numSteps:${perf.stateTrace.length}," +
          s"fEval:${perf.numObjEval},dfEval:${perf.numGradEval}")
      case _ => sys.error("No result found!")
    }
  }
}
