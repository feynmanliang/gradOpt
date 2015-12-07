package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._

// A bracketing interval where f(mid) < f(lb) and f(mid) < f(ub), guaranteeing a minimum
private[gradopt] case class BracketInterval(lb: Double, mid: Double, ub: Double) {
  def contains(x: Double): Boolean = lb <= x && ub >= x
  def size: Double = ub - lb
}

class Optimizer {
  /**
  * Minimize a convex scalar function `f` with derivative `df` and initial
  * guess `x0`.
  */
  def minimize(
      f: Double => Double, df: Double => Double, x0: Double, tol: Double = 1E-8): Option[Double] = {
    // Stream of x values returned by bracket/line search algorithm
    def improve(x: Double): Stream[Double] = {
      bracket(f, x) match {
        case Some(BracketInterval(lb,mid,ub)) => {
          // line search on bracket half which gradient points against
          val halfBkt = if (df(x) > 0) {
            BracketInterval(lb,mid,mid)
          } else {
            BracketInterval(mid,mid,ub)
          }
          val xnew = lineSearch(f, x, halfBkt)
          x #:: improve(xnew)
        }
        case None => x #:: Stream.Empty
      }
    }

    improve(x0)
      .take(5000) // limit to 5000 iteration, TODO: better stopping criterion
      .sliding(2)
      .find(_ match {
        case x#::y#::xs => math.abs(f(y) - f(x)) < tol
        case _ => false
      })
      .map(_.drop(1).head)
  }

  /**
  * Brackets the minimum of a scalar function `f`. This function uses `x0` as
  * the midpoint around which to identify the bracket bounds.
  */
  private[gradopt] def bracket(f: Double => Double, x0: Double): Option[BracketInterval] = {
    def doubleBounds(currBracket: BracketInterval): Stream[BracketInterval] =
      currBracket match {
      case BracketInterval(lb, mid, ub) => {
        val delta = mid - lb
        val newLb = if (f(mid) < f(lb)) lb else lb - delta
        val newUb = if (f(mid) < f(ub)) ub else ub + delta
        currBracket #:: doubleBounds(BracketInterval(newLb, mid, newUb))
      }
    }

    val initBracket = BracketInterval(x0 - 1D, x0, x0 + 1D)

    doubleBounds(initBracket)
    .take(5000) // stops after 5000 brackets, TODO: more robust stopping criterion
    .find(_ match {
        case BracketInterval(lb, mid, ub) => f(lb) > f(mid) && f(ub) > f(mid)
      })
  }

  /**
  * Performs a line search for x' = x + a*p within a bracketing interval to determine step size.
  * Returns the value x' which minimizes `f` along the line search.
  * This method linearly interpolates the bracket interval and chooses the minimizer of f.
  * TODO: bisection search the candidates
  */
  private[gradopt] def lineSearch(
      f: Double => Double, x: Double, bracket: BracketInterval): Double = {
    val numPoints = 100D // TODO: increase this if bracketing doesn't improve
    val candidates = x +: (bracket.lb to bracket.ub by bracket.size/numPoints)
    candidates.minBy(f)
  }
}

/**
* Client for Optimizer implementing coursework questions
*/
object Optimizer {
  def q2(): Unit = {
    val f = (x: Double) => pow(x,4D) * cos(1D/x) + 2D * pow(x,4D)
    val df = (x: Double) => 8D * pow(x,3D) + 4D * pow(x, 3D) * cos(x) - pow(x, 4D) * sin(x)
    println(f(10))
    println(df(10))
  }
  def main(args: Array[String]) = {
    q2()
  }
}

// vim: set ts=2 sw=2 et sts=2:
