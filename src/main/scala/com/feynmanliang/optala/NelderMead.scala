package com.feynmanliang.optala

import breeze.linalg._

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

/**
* @param tol the minimum change in decision variable norm to continue
*/
class NelderMeadOptimizer(
    var maxSteps: Int = 50000,
    var tol: Double = 1E-6) {

  /**
  * Initializes Nedler Mead with random `n`-point simplex in `d` dimensions,
  * each entry ~ U[-1,1].
  */
  def minimize(
      f: Vector[Double] => Double,
      d: Int,
      n: Int,
      reportPerf: Boolean): (Option[Vector[Double]], Option[PerfDiagnostics[Simplex]]) = {
    val init = Simplex(Seq.fill(n) {
      val x = 2D * (DenseVector.rand(d) - DenseVector.fill(d){0.5})
      (x, f(x))
    })
    minimize(f, init, reportPerf)
  }

  /**
  * Performs Nelder-Mead starting with initial simplex `init`.
  * TODO: return and plot simplex evolution by parameterizing PerfDiagnostics
  */
  def minimize(
      f: Vector[Double] => Double,
      init: Simplex,
      reportPerf: Boolean): (Option[Vector[Double]], Option[PerfDiagnostics[Simplex]]) = {
    require(init.points.size >= 3, "must have at least 3 points in simplex")
    val fCnt = new FunctionWithCounter(f)

    val xValues = nelderMead(fCnt, init)
      .take(maxSteps)
      .map((s:Simplex) => (s, s.points.map(_._2).sum / (s.points.size * 1D)))
      .iterator

    if (reportPerf) {
      val xValuesSeq = xValues.toSeq
      val res = xValuesSeq.sliding(2).find(x => norm(x(1)._2 - x.head._2) < tol).map(_(1))
      val trace = res match {
        case Some(xStar) => xValuesSeq
          .sliding(2)
          .takeWhile(x => norm(x(1)._2 - x.head._2) >= tol)
          .map(_(0)._1)
          .toSeq :+ xStar._1
        case None => xValuesSeq
          .sliding(2)
          .takeWhile(x => norm(x(1)._2 - x.head._2) >= tol)
          .map(_(0)._1)
      }
      val perf = PerfDiagnostics(trace.toList, fCnt.numCalls, 0)
      (res.map(s => s._1.points.map(_._1).reduce(_+_) / (1D * s._1.points.size)), Some(perf))
    } else {
      val res = xValues.sliding(2).find(x => norm(x(1)._2 - x.head._2) < tol).map(_(1))
      (res.map(s => s._1.points.map(_._1).reduce(_+_) / (1D * s._1.points.size)), None)
    }
  }

  private def nelderMead(
    f: Vector[Double] => Double,
    simplex: Simplex): Stream[Simplex] = {
    val nPts = simplex.sortedPoints.init
    val (x1, fx1) = simplex.sortedPoints.head
    val (xn, fxn) = nPts.last
    val (xnp1, fxnp1) = simplex.sortedPoints.last

    val xRefl = simplex.xBar(-1D)
    val fRefl = f(xRefl)

    if (fx1 <= fRefl && fRefl < fxn) {
      // reflected point neither best nor xnp1
      simplex #:: nelderMead(f, Simplex(nPts :+ (xRefl, fRefl)))
    } else if (fRefl < fx1) {
      // reflected point is best, go further
      val xRefl2 = simplex.xBar(-2D)
      val fRefl2 = f(xRefl2)
      if (fRefl2 < fRefl) {
        simplex #:: nelderMead(f, Simplex(nPts :+ (xRefl2, fRefl2)))
      } else {
        simplex #:: nelderMead(f, Simplex(nPts :+ (xRefl, fRefl)))
      }
    } else {
      // reflected point worse than x_n, contract
      val xReflOut = simplex.xBar(-.5D)
      val fReflOut = f(xReflOut)
      if (fxn <= fRefl && fRefl < fxnp1 && fReflOut <= fRefl) {
        // try ``outside'' contraction
        simplex #:: nelderMead(f, Simplex(nPts :+ (xReflOut, fReflOut)))
      } else{
        val xReflIn = simplex.xBar(.5D)
        val fReflIn = f(xReflIn)
        if (fReflIn < fxnp1) {
          // try ``inside'' contraction
          simplex #:: nelderMead(f, Simplex(nPts :+ (xReflIn, fReflIn)))
        } else {
          // neither outside nor inside contraction acceptable, shrink simplex towards x1
          simplex #:: nelderMead(f, Simplex(simplex.points.map { x =>
            val newX = .5D * (x1 + x._1)
            (newX, f(newX))
          }))
        }
      }
    }
  }
}

// vim: set ts=2 sw=2 et sts=2:
