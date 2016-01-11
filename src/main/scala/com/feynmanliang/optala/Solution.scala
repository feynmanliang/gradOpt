package com.feynmanliang.optala

import breeze.linalg.{DenseVector, Vector}

/** A (possibly incorrect) solution to an optimization problem.
  * @param f objective function minimized by solution
  * @param point assignment to decision variables
  */
class Solution private[optala] (val f: (Vector[Double]) => Double, val point: DenseVector[Double]) {
  lazy val objVal: Double = f(point) // objective value at `point` i.e. f(point)
}

private[optala] object Solution {
  /** Factory method for instantiating `Solution`s */
  def apply(f: (Vector[Double]) => Double, point: DenseVector[Double]): Solution = new Solution(f, point)
}

