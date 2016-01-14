package com.feynmanliang.optala.examples

import scala.util.{Failure, Success}

import breeze.linalg.{DenseMatrix, DenseVector, Vector, norm}
import breeze.numerics.pow
import breeze.stats.distributions.Uniform

import com.feynmanliang.optala.GradientOptimizer
import com.feynmanliang.optala.GradientAlgorithm._
import com.feynmanliang.optala.LineSearchConfig._
import com.feynmanliang.optala.Solution
import com.feynmanliang.optala.examples.ExampleUtils.runWithResults
import com.feynmanliang.optala.neldermead.{NelderMead, Simplex}

object MultivariateGradientOptimizerExample {
  val SEED = 42L

  def main(args: Array[String]) {
    val f: Vector[Double] => Double = v => pow(1D - v(0), 2) + 100D * pow(v(1) - pow(v(0), 2), 2)
    val df: Vector[Double] => DenseVector[Double] = v => {
      DenseVector(
        -2D * (1 - v(0)) - 400D * v(0) * (-pow(v(0), 2) + v(1)),
        200D * (-pow(v(0), 2) + v(1))
      )
    }
    val xOpt = (1D, 1D)

    val gradOpt = new GradientOptimizer(maxSteps = 50000, tol = 1E-6)
    val nmOpt = new NelderMead(maxIter = 10000, tol = 1E-10)
    for {
      (x0,i) <- List(
        DenseVector(-3D, -4D),
        DenseVector(-3D, 6D),
        DenseVector(0D, 7D),
        DenseVector(1D, -3D)).zipWithIndex
    } {
      for (algo <- List(SteepestDescent, ConjugateGradient)) {
        runWithResults(s"optimizing Rosenbrock, $algo, $x0", SEED, s"rosenbrock-$algo-$i.csv") { _ =>
          gradOpt.minimize(f, df, x0, algo, CubicInterpolation) match {
            case Success(results) =>
              DenseMatrix.horzcat(results.stateTrace.map(x => DenseMatrix(x.normGrad +: x.point.toArray: _*)): _*)
            case Failure(e) => throw e
          }
        }
      }

      runWithResults(s"optimizing Rosenbrock function, nelder-mead, $x0", SEED, s"rosenbrock-nm-$i.csv") { seedRB =>
      val initialSimplex = Simplex(Seq.fill(8) {
          val simplexPoint = DenseVector(
            Uniform(-2D, 2D)(seedRB).sample(),
            Uniform(-2D, 2D)(seedRB).sample())
          Solution(f, simplexPoint + x0)
        })
        val result = nmOpt.minimize(f, initialSimplex)
        // columns = (x1,y1,x2,y2,...), rows = iterations
        DenseMatrix.horzcat(result.stateTrace.map { simplex =>
          val centroid = simplex.centroid
          DenseMatrix(norm(df(centroid)) +: centroid.toArray: _*)
        }: _*)
      }
    }
  }
}
