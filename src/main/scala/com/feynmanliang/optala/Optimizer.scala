package com.feynmanliang.optala

import breeze.linalg.{DenseVector, Vector}

/** A point in a vector space and the value of the objective function evaluated at that point*/
private[optala] class Solution(val f: (Vector[Double]) => Double, val point: DenseVector[Double]) {
  lazy val objVal: Double = f(point)
}

private[optala] object Solution {
  def apply(f: (Vector[Double]) => Double, point: DenseVector[Double]): Solution = new Solution(f, point)
}

/** The results from a run of an iterative optimization algorithm along with performance diagnostic information. */
trait RunResult[T] {
  val bestSolution: Solution // (solution, objectiveValue) of best solution found
  val stateTrace: List[T] // sequence of states across iterations
  val numObjEval: Long // number of objective function evaluations until termination
  val numGradEval: Long // number of gradient evaluations until termination
}

/**
  * Wrapper for a function which counts the number of times it is called.
  * @param f the function to wrap
  * @tparam T the input type of the function
  * @tparam U the output type of the function
  */
private[optala] class FunctionWithCounter[-T, +U](f: T => U) extends Function[T, U] {
  var numCalls: Int = 0

  override def apply(t: T): U = {
    numCalls += 1
    f(t)
  }
}
