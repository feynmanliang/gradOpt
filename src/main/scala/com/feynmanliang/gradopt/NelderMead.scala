package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._

/** Nelder-Mead simplex, refined at each iteration */
case class Simplex(points: Seq[(Vector[Double], Double)]) {
  val sortedPoints = points.sortBy(_._2)
  val n = sortedPoints.size - 1D

  /** Centroid of the first n points */
  val nCentroid: Vector[Double] = sortedPoints.init.map(_._1).reduce(_+_) / n

  /** Ray from n-centroid to n+1st point */
  def xBar(t: Double): Vector[Double] = {
    val xnp1 = sortedPoints.last._1
    nCentroid + t * (xnp1 - nCentroid)
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
      val nPts = simplex.sortedPoints.init
      val (x1, fx1) = simplex.sortedPoints.head
      val (xn, fxn) = nPts.last
      val (xnp1, fxnp1) = simplex.sortedPoints.last

      val xRefl = simplex.xBar(-1D)
      val fRefl = f(xRefl)

      if (fx1 <= fRefl && fRefl < fxn) {
        // reflected point neither best nor xnp1
        simplex #:: next(Simplex(nPts :+ (xRefl, fRefl)))
      } else if (fRefl < fx1) {
        // reflected point is best, go further
        val xRefl2 = simplex.xBar(-2D)
        val fRefl2 = f(xRefl2)
        if (fRefl2 < fRefl) {
          simplex #:: next(Simplex(nPts :+ (xRefl2, fRefl2)))
        } else {
          simplex #:: next(Simplex(nPts :+ (xRefl, fRefl)))
        }
      } else {
        // reflected point worse than x_n, contract
        val xReflOut = simplex.xBar(-.5D)
        val fReflOut = f(xReflOut)
        val xReflIn = simplex.xBar(.5D)
        val fReflIn = f(xReflIn)
        if (fxn <= fRefl && fRefl < fxnp1 && fReflOut <= fRefl) {
          // try ``outside'' contraction
          simplex #:: next(Simplex(nPts :+ (xReflOut, fReflOut)))
        } else if (fReflIn < fxnp1) {
          // try ``inside'' contraction
          simplex #:: next(Simplex(nPts :+ (xReflIn, fReflIn)))
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
