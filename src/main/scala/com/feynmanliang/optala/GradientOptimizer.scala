package com.feynmanliang.optala

import scala.util.{Failure, Success, Try}

import breeze.linalg._

private[optala] case class GradientBasedSolution(
    override val point: DenseVector[Double],
    override val f: Vector[Double] => Double,
    grad: DenseVector[Double],
    normGrad: Double) extends Solution(f, point)

private[optala] case class GradientBasedRunResult (
    override val stateTrace: List[GradientBasedSolution],
    override val numObjEval: Long,
    override val numGradEval: Long) extends RunResult[GradientBasedSolution] {
  override val bestSolution = stateTrace.minBy(_.objVal)
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
      lineSearchConfig: LineSearchConfig.LineSearchConfig): Try[GradientBasedRunResult] = {
    val fCnt = new FunctionWithCounter[Vector[Double], Double](x => 0.5D * (x.t * (A * x)) - b.t * x)
    val dfCnt = new FunctionWithCounter[Vector[Double], Vector[Double]](x => A * x - b)

    val lineSearch: (Vector[Double], Vector[Double]) => Option[Double] = lineSearchConfig match {
      case Exact => (x, p) => LineSearch.exactLineSearch(A, dfCnt(x), x, p)
      case CubicInterpolation => (x, p) => LineSearch.chooseStepSize(fCnt, dfCnt, x, p)
    }
    val xValues = (gradientAlgorithm match {
      case SteepestDescent => steepestDescent(lineSearch, fCnt, dfCnt, x0.toDenseVector)
      case ConjugateGradient => conjugateGradient(lineSearch, fCnt, dfCnt, x0.toDenseVector)
    }).take(maxSteps)
      .toSeq

    xValues.find(_.normGrad < tol) match {
      case Some(xStar) =>
        val trace = (xValues.takeWhile(_.normGrad >= tol) :+ xStar).toList
        Success(GradientBasedRunResult(trace, fCnt.numCalls, dfCnt.numCalls))
      case None => Failure(sys.error("Did not converge!"))
    }
  }

  // Overload which vectorizes scalar-valued functions.
  def minimize(
      f: Double => Double,
      df: Double => Double,
      x0: Double,
      gradientAlgorithm: GradientAlgorithm.GradientAlgorithm,
      lineSearchConfig: LineSearchConfig.LineSearchConfig): Try[GradientBasedRunResult] = {
    val vecF: Vector[Double] => Double = v => {
      require(v.size == 1, s"vectorized f expected dimension 1 input but got ${v.size}")
      f(v(0))
    }
    val vecDf: Vector[Double] => Vector[Double] = v => {
      require(v.size == 1, s"vectorized f expected dimension 1 input but got ${v.size}")
      DenseVector(df(v(0)))
    }
    minimize(vecF, vecDf, DenseVector(x0), gradientAlgorithm, lineSearchConfig)
  }

  /**
  * Minimize a convex function `f` with derivative `df` and initial
  * guess `x0`.
  */
  def minimize(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x0: Vector[Double],
      gradientAlgorithm: GradientAlgorithm,
      lineSearchConfig: LineSearchConfig): Try[GradientBasedRunResult] = {
    val fCnt = new FunctionWithCounter(f)
    val dfCnt = new FunctionWithCounter(df)

    val lineSearch: (Vector[Double], Vector[Double]) => Option[Double] = (x, p) => {
      LineSearch.chooseStepSize(fCnt, dfCnt, x, p)
    }

    val xValues = (gradientAlgorithm match {
      case SteepestDescent => steepestDescent(lineSearch, fCnt, dfCnt, x0.toDenseVector)
      case ConjugateGradient => conjugateGradient(lineSearch, fCnt, dfCnt, x0.toDenseVector)
    }).take(maxSteps)
      .toSeq

    xValues.find(_.normGrad < tol) match {
      case Some(xStar) =>
        val trace = (xValues.takeWhile(_.normGrad >= tol) :+ xStar).toList
        Success(GradientBasedRunResult(trace, fCnt.numCalls, dfCnt.numCalls))
      case None => Failure(sys.error("Did not converge!"))
    }
  }

  /** Steepest Descent */
  private def steepestDescent(
      lineSearch: (Vector[Double], Vector[Double]) => Option[Double],
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x0: DenseVector[Double]): Stream[GradientBasedSolution] = {
    /** Computes a Stream of x values along steepest descent direction */
    def improve(x: DenseVector[Double]): Stream[GradientBasedSolution] = {
      val grad = df(x).toDenseVector
      val currSolution = GradientBasedSolution(x, f, grad, norm(grad))
      if (currSolution.normGrad == 0D) {
        currSolution #:: Stream.Empty
      } else {
        val p = -grad / norm(grad.toDenseVector) // steepest descent direction
        lineSearch(x, p) match {
          case Some(alpha) => currSolution #:: improve(x + alpha * p)
          case None => currSolution #:: Stream.Empty
        }
      }
    }
    improve(x0)
  }

  /** Conjugate Gradient using Fletcher-Reeves rule. */
  private def conjugateGradient(
      lineSearch: (Vector[Double], Vector[Double]) => Option[Double],
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x0: DenseVector[Double]): Stream[GradientBasedSolution] = {
    /** Compute a Stream of x values using CG minimizing `f`. */
    def improve(
        x: DenseVector[Double],
        grad: DenseVector[Double],
        p: DenseVector[Double]): Stream[GradientBasedSolution] = {
      val currSolution = GradientBasedSolution(x, f, grad, norm(grad))
      if (currSolution.normGrad == 0) {
        currSolution #:: Stream.Empty
      } else {
        lineSearch(x, p) match {
          case Some(alpha) =>
            val newX = x + alpha * p
            val newGrad = df(newX).toDenseVector
            val beta = (newGrad dot newGrad) / (grad dot grad)
            val newP = -newGrad + beta * p
            currSolution #:: improve(newX, newGrad, newP)
          case None => currSolution #:: Stream.Empty
        }
      }
    }
    val dfx0 = df(x0).toDenseVector
    improve(x0, dfx0, -dfx0)
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
