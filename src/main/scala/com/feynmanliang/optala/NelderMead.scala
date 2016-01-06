package com.feynmanliang.optala

import breeze.linalg.{norm, Vector, DenseVector}

/** A `n+1` point simplex */
private[optala] case class Simplex(private val points: Seq[Solution]) {
  val sortedSolutions = points.sortBy(_.objVal)
  val n = sortedSolutions.size - 1D

  /** Simplex point minimizing objective function */
  val bestSolution = sortedSolutions.head

  /** Centroid of all `n+1` points */
  val centroid: DenseVector[Double] = sortedSolutions.map(_.point).reduce(_+_) / (n+1)

  /** Centroid of the first n points */
  private[optala] val nCentroid: DenseVector[Double] = sortedSolutions.init.map(_.point).reduce(_+_) / n

  /** Ray from `nCentroid` to n+1 point */
  private[optala] def xBar(t: Double): DenseVector[Double] = {
    val xNplus1 = sortedSolutions.last.point
    nCentroid + t * (xNplus1 - nCentroid)
  }

  /** Average objective value over all n+1 simplex points */
  private[optala] val averageObjVal: Double = sortedSolutions.map(_.objVal).sum / (n+1)
}

private[optala] case class NelderMeadRunResult(
    override val stateTrace: List[Simplex],
    override val numObjEval: Long,
    override val numGradEval: Long) extends RunResult[Simplex] {
  override val bestSolution = stateTrace.maxBy(_.bestSolution.objVal).bestSolution
}

/** An implementation of the Nelder-Mead optimization method.
  *
  * @param maxObjEvals maximum number of objective function evaluations before termination
  * @param maxIter maximum number of iterations before termination
  * @param tol minimum change in norm between simplex centroids from two consecutive iterations before termination
  */
class NelderMeadOptimizer(
    var maxObjEvals: Int = Int.MaxValue,
    var maxIter: Int = 50000,
    var tol: Double = 1E-6) {

  /** Randomly initialize the initial simplex and run Nelder-Mead.
    *
    * @param f objective function to be minimized
    * @param d dimensionality of the input domain of `f`
    * @param n size of Nelder-Mead simplex
    */
  def minimize(
      f: Vector[Double] => Double,
      d: Int,
      n: Int): NelderMeadRunResult = {
    val init = Simplex(Seq.fill(n) {
      val x = 2D * (DenseVector.rand(d) - DenseVector.fill(d){0.5}) // randomly generate points in [-1,1]
      Solution(f, x)
    })
    minimize(f, init)
  }

  /** Run Nelder-Mead to minimize a function without requiring its gradient.
    *
    * @param f objective function to be minimized
    * @param init initial simplex
    * @return
    */
  def minimize(
      f: Vector[Double] => Double,
      init: Simplex): NelderMeadRunResult = {
    require(init.n + 1 >= 2, "must have at least 2 points in simplex")
    val fCnt = new FunctionWithCounter(f)

    val xValues = nelderMead(fCnt, init)
      .takeWhile(_ => fCnt.numCalls <= maxObjEvals)
      .take(maxIter)
    val trace = xValues.head +: xValues
      .sliding(2)
      .takeWhile(x => norm(x(1).centroid - x.head.centroid) >= tol)
      .map(_(1))
      .toSeq
    NelderMeadRunResult(trace.toList, fCnt.numCalls, 0)
  }

  private def nelderMead(
    f: Vector[Double] => Double,
    simplex: Simplex): Stream[Simplex] = {
    val x1toN = simplex.sortedSolutions.init
    val x1 = simplex.sortedSolutions.head
    val xN = x1toN.last
    val xNplus1 = simplex.sortedSolutions.last

    val xRefl = Solution(f, simplex.xBar(-1D))
    if (x1.objVal <= xRefl.objVal && xRefl.objVal < xN.objVal) {
      // reflected point neither best nor xnp1
      simplex #:: nelderMead(f, Simplex(x1toN :+ xRefl))
    } else if (xRefl.objVal < x1.objVal) {
      // reflected point is best, go further
      val xRefl2 = Solution(f, simplex.xBar(-2D))
      if (xRefl2.objVal < xRefl.objVal) {
        simplex #:: nelderMead(f, Simplex(x1toN :+ xRefl2))
      } else {
        simplex #:: nelderMead(f, Simplex(x1toN :+ xRefl))
      }
    } else {
      // reflected point worse than x_n, contract
      val xReflOut = Solution(f, simplex.xBar(-.5D))
      if (xN.objVal <= xRefl.objVal && xRefl.objVal < xNplus1.objVal && xReflOut.objVal <= xRefl.objVal) {
        // try ``outside'' contraction
        simplex #:: nelderMead(f, Simplex(x1toN :+ xReflOut))
      } else{
        val xReflIn = Solution(f, simplex.xBar(.5D))
        if (xReflIn.objVal < xNplus1.objVal) {
          // try ``inside'' contraction
          simplex #:: nelderMead(f, Simplex(x1toN :+ xReflIn))
        } else {
          // neither outside nor inside contraction acceptable, shrink simplex towards x1
          simplex #:: nelderMead(f, Simplex(simplex.sortedSolutions.map { x =>
            val newX = .5D * (x1.point + x.point)
            Solution(f, newX)
          }))
        }
      }
    }
  }
}
