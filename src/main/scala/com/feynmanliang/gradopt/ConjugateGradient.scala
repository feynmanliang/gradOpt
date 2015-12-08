package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._

/**
* Minimizes a function using conjugate gradient.
*/
class ConjugateGradient() {
  /**
  * Nocedal Algorithm 5.2
  * TODO: save the trace
  **/
  def minimize(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x0: Vector[Double],
      reportPerf: Boolean): Option[Vector[Double]] = {

    val fCnt = new FunctionWithCounter(f)
    val dfCnt = new FunctionWithCounter(df)

    def improve(
        x: Vector[Double],
        grad: Vector[Double],
        p: Vector[Double]): Stream[(Vector[Double], Double)] =
      LineSearch.chooseStepSize(f, p, df, x) match {
        case Some(alpha) => {
          val newX = x + alpha * p
          val newGrad = df(newX)
          val beta = (newGrad dot newGrad) / (grad dot grad) // Fletcher-Reeves rule
          val newP = -newGrad + beta * p
          (x, norm(grad.toDenseVector)) #:: improve(newX, newGrad, newP)
        }
        case None => (x, norm(grad.toDenseVector)) #:: Stream.Empty
      }

    val xValues = {
      val dfx0 = df(x0)
      improve(x0, dfx0, -dfx0)
    }.iterator

    xValues.find(_._2 < 1E-6).map(_._1)
  }
}
