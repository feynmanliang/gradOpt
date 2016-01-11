package com.feynmanliang.optala.examples

import java.io.File

import breeze.linalg.{DenseVector, Matrix, Vector, csvwrite}
import breeze.stats.distributions.Uniform
import com.feynmanliang.optala.Solution
import com.feynmanliang.optala.neldermead.Simplex

/** Utility functions for example code and experiments. */
private[examples] object ExampleUtils {
  def experimentWithResults(
      experimentName: String,
      resultFName: String)(results: => Matrix[Double]) = {
    println(s"=== Performing experiment: $experimentName ===")
    val resultsFile = new File(s"results/$resultFName")
    println(s"Writing results to: $resultsFile")
    csvwrite(resultsFile, results)
  }

  def createRandomSimplex(n: Int, f: Vector[Double] => Double): Simplex = Simplex(Seq.fill(n) {
    val simplexPoint = DenseVector(Uniform(-2D, 2D).sample(), Uniform(-1D, 1D).sample())
    Solution(f, simplexPoint)
  })
}
