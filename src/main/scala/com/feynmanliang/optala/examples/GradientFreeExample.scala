package com.feynmanliang.optala.examples

import java.io.File

import org.apache.commons.math3.random.MersenneTwister

import breeze.linalg.{DenseMatrix, DenseVector, Vector, csvwrite}
import breeze.numerics.pow
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator, Uniform}

import com.feynmanliang.optala.{SelectionStrategy, GeneticAlgorithm, NelderMeadOptimizer, Simplex}

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

    println(s"===Optimizing 6HCF using Nedler-Mead===")
    val nmOpt = new NelderMeadOptimizer(maxSteps = 1000, tol = 1E-10)
    val initialSimplex = Simplex(Seq.fill(8){
      val simplexPoint = DenseVector(Uniform(-2D,2D).sample(), Uniform(-1D,1D).sample())
      (simplexPoint, f(simplexPoint))
    })
    nmOpt.minimize(f, initialSimplex, reportPerf = true) match {
      case (_, Some(perf)) =>
        val sStar = perf.stateTrace.last
        val xStar = sStar.points.minBy(_._2)._1
        val fStar = f(xStar)

        // columns = (x1,y1,x2,y2,...), rows = iterations
        val stateTrace = DenseMatrix.vertcat(perf.stateTrace.map { simplex =>
          DenseMatrix(simplex.points.flatMap(_._1.toArray))
        }: _*)
        val stateTraceFile = new File("results/nm-stateTrace.csv")
        csvwrite(stateTraceFile, stateTrace)
        println(s"Wrote stateTrace to $stateTraceFile")

         // objective value at best point on simplex
        val objTrace = DenseMatrix(perf.stateTrace.map(_.points.map(_._2).min): _*)
        val objTraceFile = new File("results/nm-objTrace.csv")
        csvwrite(objTraceFile, objTrace)
        println(s"Wrote objTrace to $objTraceFile")

        println(s"xStar: $xStar\n" +
          s"fStar:$fStar\n" +
          s"numIters:${perf.stateTrace.length}\n" +
          s"numObjEval: ${perf.numObjEval}\n" +
          s"numGradEval:${perf.numGradEval}")
      case _ => sys.error("No result found!")
    }

    println(s"===Optimizing 6HCF using GA===")
    val ga = new GeneticAlgorithm(maxSteps = 1000)
    val popSize = 20
    val eliteCount = 2
    val xoverFrac = 0.8
    ga.minimize(f, lb, ub, popSize, SelectionStrategy.TournamentSelection, eliteCount, xoverFrac, Some(seed)) match {
      case (_, Some(perf)) =>
        val xstar = perf.stateTrace.last.population.minBy(_._2)._1
        val fstar = f(xstar)

        // columns = (x1,y1,x2,y2,...), rows = iterations
        val stateTrace = DenseMatrix.vertcat(perf.stateTrace.map { gen =>
          DenseMatrix(gen.population.flatMap(_._1.toArray))
        }: _*)
        val stateTraceFile = new File("results/ga-stateTrace.csv")
        csvwrite(stateTraceFile, stateTrace)
        println(s"Wrote stateTrace to $stateTraceFile")

        // objective value at best point on simplex
        val objTrace = DenseMatrix(perf.stateTrace.map(_.population.map(_._2).min): _*)
        val objTraceFile = new File("results/ga-objTrace.csv")
        csvwrite(objTraceFile, objTrace)
        println(s"Wrote objTrace to $objTraceFile")

        println(s"popSize: $popSize\n" +
          s"eliteCount: $eliteCount\n" +
          s"xstar:$xstar\n" +
          s"fstar:$fstar\n" +
          s"numSteps:${perf.stateTrace.length}\n" +
          s"fEval:${perf.numObjEval}\n" +
          s"dfEval:${perf.numGradEval}")
      case _ => sys.error("No result found!")
    }
  }
}
