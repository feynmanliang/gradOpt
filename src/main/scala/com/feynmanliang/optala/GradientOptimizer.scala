package com.feynmanliang.optala

import breeze.linalg._
import breeze.numerics._

// A bracketing interval where f(x + mid'*df) < f(x + lb'*df) and f(x + mid'*df) < f(x + ub'*df),
// guaranteeing a minimum
private[optala] case class BracketInterval(lb: Double, mid: Double, ub: Double) {
  def contains(x: Double): Boolean = lb <= x && ub >= x
  def size: Double = ub - lb
}

class GradientOptimizer(
    var maxSteps: Int = 50000,
    var tol: Double = 1E-6) {
  import com.feynmanliang.optala.GradientAlgorithm._
  import com.feynmanliang.optala.LineSearchConfig._

  /** Minimizes a quadratic form 0.5 x'Ax - b'x using exact step size */
  def minQuadraticForm(
      A: Matrix[Double],
      b: Vector[Double],
      x0: Vector[Double],
      gradientAlgorithm: GradientAlgorithm.GradientAlgorithm,
      lineSearchConfig: LineSearchConfig.LineSearchConfig,
      reportPerf: Boolean): (Option[Vector[Double]], Option[PerfDiagnostics[Vector[Double]]]) = {

    val fCnt = new FunctionWithCounter[Vector[Double], Double](x => 0.5D * (x.t * (A * x)) - b.t * x)
    val dfCnt = new FunctionWithCounter[Vector[Double], Vector[Double]](x => A * x - b)

    val lineSearch: (Vector[Double], Vector[Double]) => Option[Double] = lineSearchConfig match {
      case Exact => (x, p) => LineSearch.exactLineSearch(A, dfCnt(x), x, p)
      case CubicInterpolation => (x, p) => LineSearch.chooseStepSize(fCnt, dfCnt, x, p)
    }
    val xValues = (gradientAlgorithm match {
      case SteepestDescent => steepestDescent(lineSearch, dfCnt, x0)
      case ConjugateGradient => conjugateGradient(lineSearch, dfCnt, x0)
    }).take(maxSteps).iterator

    if (reportPerf) {
      val xValuesSeq = xValues.toSeq
      val res = xValuesSeq.find(_._2 < tol)
      val trace = res match {
        case Some(xStar) => xValuesSeq.takeWhile(_._2 >= tol) :+ xStar
        case None => xValuesSeq.takeWhile(_._2 >= tol)
      }
      val perf = PerfDiagnostics(trace.toList, fCnt.numCalls, dfCnt.numCalls)
      (res.map(_._1), Some(perf))
    } else {
      val res = xValues.find(_._2 < tol).map(_._1)
      (res, None)
    }

  }

  // Overload which vectorizes scalar-valued functions.
  def minimize(
      f: Double => Double,
      df: Double => Double,
      x0: Double,
      gradientAlgorithm: GradientAlgorithm.GradientAlgorithm,
      lineSearchConfig: LineSearchConfig.LineSearchConfig,
      reportPerf: Boolean): (Option[Vector[Double]], Option[PerfDiagnostics[Vector[Double]]]) = {
    val vecF: Vector[Double] => Double = v => {
      require(v.size == 1, s"vectorized f expected dimension 1 input but got ${v.size}")
      f(v(0))
    }
    val vecDf: Vector[Double] => Vector[Double] = v => {
      require(v.size == 1, s"vectorized f expected dimension 1 input but got ${v.size}")
      DenseVector(df(v(0)))
    }
    minimize(vecF, vecDf, DenseVector(x0), gradientAlgorithm, lineSearchConfig, reportPerf)
  }

  /**
  * Minimize a convex function `f` with derivative `df` and initial
  * guess `x0`. Uses steepest-descent.
  */
  def minimize(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x0: Vector[Double],
      gradientAlgorithm: GradientAlgorithm,
      lineSearchConfig: LineSearchConfig,
      reportPerf: Boolean): (Option[Vector[Double]], Option[PerfDiagnostics[Vector[Double]]]) = {

    val fCnt = new FunctionWithCounter(f)
    val dfCnt = new FunctionWithCounter(df)

    val lineSearch: (Vector[Double], Vector[Double]) => Option[Double] = (x, p) => {
      LineSearch.chooseStepSize(fCnt, dfCnt, x, p)
    }

    val xValues = (gradientAlgorithm match {
      case SteepestDescent => steepestDescent(lineSearch, dfCnt, x0)
      case ConjugateGradient => conjugateGradient(lineSearch, dfCnt, x0)
    }).take(maxSteps).iterator


    if (reportPerf) {
      val xValuesSeq = xValues.toSeq
      val res = xValuesSeq.find(_._2 < tol)
      val trace = res  match {
        case Some(xStar) => xValuesSeq.takeWhile(_._2 >= tol) :+ xStar
        case None => xValuesSeq.takeWhile(_._2 >= tol)
      }
      val perf = PerfDiagnostics(trace.toList, fCnt.numCalls, dfCnt.numCalls)
      (res.map(_._1), Some(perf))
    } else {
      val res = xValues.find(_._2 < tol).map(_._1)
      (res, None)
    }
  }

  /** Steepest Descent */
  private def steepestDescent(
      lineSearch: (Vector[Double], Vector[Double]) => Option[Double],
      df: Vector[Double] => Vector[Double],
      x0: Vector[Double]): Stream[(Vector[Double], Double)] = {
    /** Computes a Stream of x values along steepest descent direction */
    def improve(x: Vector[Double]): Stream[(Vector[Double], Double)] = {
      val grad = df(x)
      val p = -grad // steepest descent direction
      lineSearch(x, p) match {
        case Some(alpha) => {
          val xnew = x + alpha * p
          (x, norm(grad.toDenseVector)) #:: improve(xnew)
        }
        case None => (x, norm(grad.toDenseVector)) #:: Stream.Empty
      }
    }
    improve(x0)
  }

  /** Conjugate Gradient using Fletcher-Reeves rule. */
  private def conjugateGradient(
      lineSearch: (Vector[Double], Vector[Double]) => Option[Double],
      df: Vector[Double] => Vector[Double],
      x0: Vector[Double]): Stream[(Vector[Double], Double)] = {
    /** Compute a Stream of x values using CG minimizing `f`. */
    def improve(
        x: Vector[Double],
        grad: Vector[Double],
        p: Vector[Double]): Stream[(Vector[Double], Double)] = {
      lineSearch(x, p) match {
        case Some(alpha) => {
          val newX = x + alpha * p
          val newGrad = df(newX)
          val beta = (newGrad dot newGrad) / (grad dot grad) // Fletcher-Reeves rule
          val newP = -newGrad + beta * p
          (x, norm(grad.toDenseVector)) #:: improve(newX, newGrad, newP)
        }
        case None => (x, norm(grad.toDenseVector)) #:: Stream.Empty
      }
    }
    val dfx0 = df(x0)
    improve(x0, dfx0, -dfx0) // initialize p to be steepest descent direction
  }
}

object GradientAlgorithm extends Enumeration {
  type GradientAlgorithm = Value
  val SteepestDescent = Value("Steepest Descent")
  val ConjugateGradient = Value("Conjugate Gradient")
}

object LineSearchConfig extends Enumeration {
  type LineSearchConfig = Value
  val CubicInterpolation = Value("Cubic Interpolation")
  val Exact = Value("Exact Line Search")
}

// vim: set ts=2 sw=2 et sts=2:
