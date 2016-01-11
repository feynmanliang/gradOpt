package com.feynmanliang.optala.examples

import java.io.File

import breeze.linalg.{DenseVector, Matrix, Vector, csvwrite}
import breeze.stats.distributions.{ThreadLocalRandomGenerator, RandBasis, Uniform}
import com.feynmanliang.optala.Solution
import com.feynmanliang.optala.neldermead.Simplex
import org.apache.commons.math3.random.MersenneTwister

/** Utility functions for example code and experiments. */
private[examples] object ExampleUtils {
  /** Runs an experiment and writes the results to a CSV. */
  def experimentWithResults(
      experimentName: String,
      seed: Long,
      resultFName: String)(results: RandBasis => Matrix[Double]) = {
    // sets random seed for this experiment
    val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

    println(s"=== Performing experiment: $experimentName ===")
    val resultsFile = new File(s"results/$resultFName")
    println(s"Writing results to: $resultsFile")
    csvwrite(resultsFile, results(rand))
  }

  /** Creates a random `n` point simplex over [-2,2] x [-1,1].
    * @param n number of simplex vertices
    * @param f objective function
    * @param rand seed for Breeze random number generator
    */
  def createRandomSimplex(n: Int, f: Vector[Double] => Double)(
      implicit rand: RandBasis): Simplex = Simplex(Seq.fill(n) {
    val simplexPoint = DenseVector(Uniform(-2D, 2D)(rand).sample(), Uniform(-1D, 1D)(rand).sample())
    Solution(f, simplexPoint)
  })
}
