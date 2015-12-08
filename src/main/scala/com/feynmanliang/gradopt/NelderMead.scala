package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._

case class Simplex(points: Seq[(Vector[Double], Double)]) {
  def centroid(): Vector[Double] = {
    points.sortBy(_._2).init.map(_._1).reduce(_+_) / (points.size - 1D)
  }

  def xBar(t: Double): Vector[Double] = {
    val c = centroid()
    val xnp1Pt = points.maxBy(_._2)._1
    c + t * (xnp1Pt - c)
  }
}

class NelderMead() {
  /**
  * Initializes Nedler Mead with random `n`-point simplex in `d` dimensions,
  * each entry ~ U[-1,1].
  */
  def minimize(f: Vector[Double] => Double, d: Int, n: Int): Stream[Simplex] = {
    val init = Simplex(Seq.fill(n) {
      val x = 2D * (DenseVector.rand(n) - DenseVector.fill(n){0.5})
      (x, f(x))
    })
    minimize(f, init)
  }

  /**
  * Performs Nelder-Mead starting with initial simplex `init`.
  */
  def minimize(f: Vector[Double] => Double, init: Simplex): Stream[Simplex] = {
    require(init.points.size >= 3, "must have at least 3 points in simplex")

    def next(simplex: Simplex): Stream[Simplex] = {
      val q = simplex.points.sortBy(_._2).init

      val (x1, fx1) = q.head
      val (xn, fxn) = q.last
      val (xnp1, fxnp1) = simplex.points.maxBy(_._2)

      val xRefl = simplex.xBar(-1D)
      val fRefl = f(xRefl)

      if (fx1 <= fRefl && fRefl < fxn) {
        // reflected point neither best nor xnp1
        simplex #:: next(Simplex(q :+ (xRefl, fRefl)))
      } else if (fRefl < fx1) {
        // reflected point is best, go further
        val xRefl2 = simplex.xBar(-2D)
        val fRefl2 = f(xRefl2)
        if (fRefl2 < fRefl) {
          simplex #:: next(Simplex(q :+ (xRefl2, fRefl2)))
        } else {
          simplex #:: next(Simplex(q :+ (xRefl, fRefl)))
        }
      } else {
        // reflected point worse than x_n, contract
        val xReflOut = simplex.xBar(-.5D)
        val fReflOut = f(xReflOut)
        val xReflIn = simplex.xBar(.5D)
        val fReflIn = f(xReflIn)
        if (fxn <= fRefl && fRefl < fxnp1 && fReflOut <= fRefl) {
          // try ``outside'' contraction
          simplex #:: next(Simplex(q :+ (xReflOut, fReflOut)))
        } else if (fReflIn < fxnp1) {
          // try ``inside'' contraction
          simplex #:: next(Simplex(q :+ (xReflIn, fReflIn)))
        } else {
          // neither outside nor inside contraction acceptable, shrink simplex towards x1
          simplex #:: next(Simplex(simplex.points.map { x =>
            val newX = .5D * (x1 + x._1)
            (newX, f(newX))
          }))
        }
      }
    }

    next(init)
  }
}

// vim: set ts=2 sw=2 et sts=2:
