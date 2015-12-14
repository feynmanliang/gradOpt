package com.feynmanliang.optala.examples

import breeze.linalg.{DenseMatrix, Vector, DenseVector}
import breeze.numerics.pow
import breeze.stats.distributions.{ThreadLocalRandomGenerator, RandBasis, Uniform}
import org.apache.commons.math3.random.MersenneTwister

import com.feynmanliang.optala.{Simplex, GradientOptimizer, NelderMeadOptimizer}
import com.feynmanliang.optala.GradientAlgorithm._
import com.feynmanliang.optala.LineSearchConfig._

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

    val gradOpt = new GradientOptimizer(maxSteps = 5000, tol = 1E-3)
    for (algo <- List(SteepestDescent, ConjugateGradient)) {
      ExampleUtils.experimentWithResults(s"optimizing Rosenbrock function using $algo", s"rosenbrock-$algo.csv") {
        gradOpt.minimize(f, df, x0, algo, CubicInterpolation, reportPerf = true) match {
          case (_, Some(perf)) =>
            println(f"$algo & ${perf.stateTrace.size} & ${perf.numObjEval} & ${perf.numGradEval}" +
              f" & ${perf.stateTrace.last._1(0)}%.3E & ${perf.stateTrace.last._2}%.3E\\\\")
            DenseMatrix.horzcat(perf.stateTrace.map(x => DenseMatrix(x._2 +: x._1.toArray: _*)): _*)
          case _ => sys.error(s"No results for x0=$x0!!!")
        }
      }
    }

    val nmOpt = new NelderMeadOptimizer(maxSteps = 5000, tol = 1E-10)
    val initialSimplex = Simplex(Seq.fill(8) {
      val simplexPoint = DenseVector(Uniform(-5D, -1D).sample(), Uniform(-6D, -2D).sample())
      (simplexPoint, f(simplexPoint))
    })
    ExampleUtils.experimentWithResults("optimizing Rosenbrock function using nelder-mead", s"rosenbrock-nm.csv") {
      nmOpt.minimize(f, initialSimplex, reportPerf = true) match {
        case (_, Some(perf)) =>
          println(f"Nelder-Mead & ${perf.stateTrace.size} & ${perf.numObjEval} & ${perf.numGradEval}" +
            f" & ${perf.stateTrace.last.points.map(_._2).min}%.3E & $$\\cdot$$ \\\\")

          // columns = (x1,y1,x2,y2,...), rows = iterations
          DenseMatrix.horzcat(perf.stateTrace.map { simplex =>
            val centroid = simplex.points.map(_._1).reduce(_+_) / simplex.points.size.toDouble
            DenseMatrix(f(centroid) +: centroid.toArray: _*)
          }: _*)
        case _ => sys.error(s"No results for x0=$x0!!!")
      }
    }
  }
}
