package com.feynmanliang.gradopt

import java.io.File

import breeze.linalg._
import breeze.numerics._
import breeze.plot._

// A bracketing interval where f(x + mid'*df) < f(x + lb'*df) and f(x + mid'*df) < f(x + ub'*df),
// guaranteeing a minimum
private[gradopt] case class BracketInterval(lb: Double, mid: Double, ub: Double) {
  def contains(x: Double): Boolean = lb <= x && ub >= x
  def size: Double = ub - lb
}

// Performance diagnostics for the optimizer
private[gradopt] case class PerfDiagnostics(
  xTrace: Seq[(Vector[Double], Double)],
  numEvalF: Long,
  numEvalDf: Long)

// TODO: these should be path-dependent to restrict to optimizer instance
private[gradopt] class FunctionWithCounter[-T,+U](f: T => U) extends Function[T,U] {
  var numCalls: Int = 0
  override def apply(t: T): U = {
    numCalls += 1
    f(t)
  }
}

class Optimizer(
    var maxSteps: Int = 50000,
    var tol: Double = 1E-6) {
  import com.feynmanliang.gradopt.GradientAlgorithm._
  import com.feynmanliang.gradopt.LineSearchConfig._

  /** Minimizes a quadratic form 0.5 x'Ax - b'x using exact step size */
  // TODO: refactor into Optimizer framework
  def minQuadraticForm(
      A: Matrix[Double],
      b: Vector[Double],
      x0: Vector[Double],
      gradientAlgorithm: GradientAlgorithm.GradientAlgorithm,
      lineSearchConfig: LineSearchConfig.LineSearchConfig,
      reportPerf: Boolean): (Option[Vector[Double]], Option[PerfDiagnostics]) = {
    val fCnt = new FunctionWithCounter[Vector[Double], Double](x => 0.5D * (x.t * (A * x)) - b.t * x)
    val dfCnt = new FunctionWithCounter[Vector[Double], Vector[Double]](x => A * x - b)

    /** Steepest Descent */
    def steepestDescent(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x0: Vector[Double]): Stream[(Vector[Double], Double)] = {
      /** Computes a Stream of x values along steepest descent direction */
      def improve(x: Vector[Double]): Stream[(Vector[Double], Double)] = {
        val grad = df(x)
        val p = -grad // steepest descent direction
        LineSearch.exactLineSearch(A, grad, x, p) match {
          case Some(alpha) => {
            val xNew = x + alpha * p
            (x, norm(grad.toDenseVector)) #:: improve(xNew)
          }
          case None => (x, norm(grad.toDenseVector)) #:: Stream.Empty
        }
      }
      improve(x0)
    }
    val xValues = (gradientAlgorithm match {
        case SteepestDescent => steepestDescent(fCnt, dfCnt, x0)
        case ConjugateGradient => ???
      }).take(maxSteps).iterator


    if (reportPerf) {
      val xValuesSeq = xValues.toSeq
      val res = xValuesSeq.find(_._2 < tol)
      val trace = res match {
        case Some(xStar) => xValuesSeq.takeWhile(_._2 >= tol) :+ xStar
        case None => xValuesSeq.takeWhile(_._2 >= tol)
      }
      val perf = PerfDiagnostics(trace, fCnt.numCalls, dfCnt.numCalls)
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
      reportPerf: Boolean): (Option[Vector[Double]], Option[PerfDiagnostics]) = {
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
      reportPerf: Boolean): (Option[Vector[Double]], Option[PerfDiagnostics]) = {

    // Wrap functions with counters to collect performance metrics
    val fCnt = new FunctionWithCounter(f)
    val dfCnt = new FunctionWithCounter(df)

    val xValues = (gradientAlgorithm match {
      case SteepestDescent => steepestDescent(fCnt, dfCnt, x0)
      case ConjugateGradient => conjugateGradient(fCnt, dfCnt, x0)
    }).take(maxSteps).iterator


    if (reportPerf) {
      val xValuesSeq = xValues.toSeq
      val res = xValuesSeq.find(_._2 < tol)
      val trace = res  match {
        case Some(xStar) => xValuesSeq.takeWhile(_._2 >= tol) :+ xStar
        case None => xValuesSeq.takeWhile(_._2 >= tol)
      }
      val perf = PerfDiagnostics(trace, fCnt.numCalls, dfCnt.numCalls)
      (res.map(_._1), Some(perf))
    } else {
      val res = xValues.find(_._2 < tol).map(_._1)
      (res, None)
    }
  }

  /** Steepest Descent */
  private def steepestDescent(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x0: Vector[Double]): Stream[(Vector[Double], Double)] = {
    /** Computes a Stream of x values along steepest descent direction */
    def improve(x: Vector[Double]): Stream[(Vector[Double], Double)] = {
      val grad = df(x)
      val p = -grad // steepest descent direction
      LineSearch.chooseStepSize(f, df, x, p) match {
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
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x0: Vector[Double]): Stream[(Vector[Double], Double)] = {
    /** Compute a Stream of x values using CG minimizing `f`. */
    def improve(
        x: Vector[Double],
        grad: Vector[Double],
        p: Vector[Double]): Stream[(Vector[Double], Double)] = {
      LineSearch.chooseStepSize(f, df, x, p) match {
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

/**
* Companion object for Optimizer.
*/
object Optimizer {
  import com.feynmanliang.gradopt.GradientAlgorithm._
  import com.feynmanliang.gradopt.LineSearchConfig._

  def q2(showPlot: Boolean = false): Unit = {
    val f = (x: Double) => pow(x,4) * cos(pow(x,-1)) + 2D * pow(x,4)
    val df = (x: Double) => 8D * pow(x,3) + 4D * pow(x, 3) * cos(pow(x,-1)) - pow(x, 4) * sin(pow(x,-1))

    if (showPlot) {
      val fig = Figure()
      val x = linspace(-.1,0.1)
      fig.subplot(2,1,0) += plot(x, x.map(f))
      fig.subplot(2,1,1) += plot(x, x.map(df))
      fig.saveas("lines.png") // save current figure as a .png, eps and pdf also supported
    }

    val opt = new Optimizer()
    for (x0 <- List(-5, -1, -0.1, -1E-2, -1E-3, -1E-4, -1E-5, 1E-5, 1E-4, 1E-3, 1E-2, 0.1, 1, 5)) {
      opt.minimize(f, df, x0, SteepestDescent, CubicInterpolation, true) match {
        case (Some(xstar), Some(perf)) =>
          println(f"x0=$x0, xstar=$xstar, numEvalF=${perf.numEvalF}, numEvalDf=${perf.numEvalDf}")
          println(perf.xTrace.toList.map("%2.4f".format(_)))
        case _ => println(s"No results for x0=$x0!!!")
      }
    }
  }

  def q3(showPlot: Boolean = false): Unit = {
    val opt = new Optimizer(maxSteps=101, tol=1E-4)
    for {
      fname <- List("A10.csv", "A100.csv", "A1000.csv", "B10.csv", "B100.csv", "B1000.csv")
    } {
      val A: DenseMatrix[Double] = csvread(new File(getClass.getResource("/" + fname).getFile()))
      assert(A.rows == A.cols, "A must be symmetric")
      val n: Int = A.cols
      val b: DenseVector[Double] = 2D * (DenseVector.rand(n) - DenseVector.fill(n){0.5})

      println(s"$fname")
      opt.minQuadraticForm(A, b, DenseVector.zeros(n), SteepestDescent, Exact, true) match {
        case (res, Some(perf)) =>
          println(s"$res, ${perf.xTrace.takeRight(2)}, ${perf.xTrace.length}, ${perf.numEvalF}, ${perf.numEvalDf}")
        case _ => throw new Exception("Minimize failed to return perf diagnostics")
      }
    }
  }

  def main(args: Array[String]) = {
    // q2(showPlot = false)
    q3(showPlot = false)
  }
}

// vim: set ts=2 sw=2 et sts=2:
