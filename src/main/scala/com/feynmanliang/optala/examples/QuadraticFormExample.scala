package com.feynmanliang.optala.examples

import java.io.File

import scala.util.{Failure, Success}

import breeze.linalg.{DenseVector, DenseMatrix, csvread}
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator, Uniform}
import org.apache.commons.math3.random.MersenneTwister

import com.feynmanliang.optala.GradientOptimizer
import com.feynmanliang.optala.GradientAlgorithm._
import com.feynmanliang.optala.LineSearchConfig._

object QuadraticFormExample {
  val SEED = 45L
  implicit val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(SEED)))

  def main(args: Array[String]) {
    val gradOpt = new GradientOptimizer(maxSteps = 100000, tol = 1E-3)
    for {
      fname <- List("A10", "A100", "A1000", "B10", "B100", "B1000")
      a = csvread(new File(args(0) + fname + ".csv"))
      n = a.cols
      b = DenseVector.fill(n) { Uniform(-1D,1D)(rand).sample() }
      lsAlgo <- List(Exact, CubicInterpolation)
      optAlgo <- List(ConjugateGradient, SteepestDescent)
    } yield {
      assert(a.rows == a.cols, "A must be symmetric")

      println(s"====$lsAlgo=====")
      gradOpt.minQuadraticForm(a, b, DenseVector.zeros(n), optAlgo, lsAlgo) match {
        case Success(results) =>
          println(s"$optAlgo & $fname & ${results.stateTrace.size} & ${results.numObjEval} & ${results.numGradEval}" +
            f" & ${results.bestSolution.objVal}%.3E & ${results.bestSolution.normGrad}%.3E\\\\")
        case Failure(e) => println(e.getMessage)
      }
    }
  }
}

