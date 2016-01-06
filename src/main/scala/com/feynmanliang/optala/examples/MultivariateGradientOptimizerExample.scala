package com.feynmanliang.optala.examples

import breeze.linalg.{DenseMatrix, DenseVector, Vector}
import breeze.numerics.pow
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import com.feynmanliang.optala.GradientAlgorithm._
import com.feynmanliang.optala.LineSearchConfig._
import com.feynmanliang.optala.examples.ExampleUtils._
import com.feynmanliang.optala.{GradientOptimizer, NelderMeadOptimizer}
import org.apache.commons.math3.random.MersenneTwister

object MultivariateGradientOptimizerExample {
  val seed = 42L
  implicit val rand = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

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

    val gradOpt = new GradientOptimizer(maxSteps = 10000, tol = 1E-6)
    for (algo <- List(SteepestDescent, ConjugateGradient)) {
      ExampleUtils.experimentWithResults(s"optimizing Rosenbrock function using $algo", s"rosenbrock-$algo.csv") {
        gradOpt.minimize(f, df, x0, algo, CubicInterpolation, reportPerf = true) match {
          case (_, Some(perf)) =>
            DenseMatrix.horzcat(perf.stateTrace.map(x => DenseMatrix(x.normGrad +: x.point.toArray: _*)): _*)
          case _ => sys.error(s"No results for x0=$x0!!!")
        }
      }
    }

    val nmOpt = new NelderMeadOptimizer(maxIter = 10000, tol = 1E-10)
    val initialSimplex = createRandomSimplex(8, f)
    ExampleUtils.experimentWithResults("optimizing Rosenbrock function using nelder-mead", s"rosenbrock-nm.csv") {
      val result = nmOpt.minimize(f, initialSimplex)
      // columns = (x1,y1,x2,y2,...), rows = iterations
      DenseMatrix.horzcat(result.stateTrace.map { simplex =>
        val centroid = simplex.centroid
        DenseMatrix(f(centroid) +: centroid.toArray: _*)
      }: _*)
    }
  }
}
