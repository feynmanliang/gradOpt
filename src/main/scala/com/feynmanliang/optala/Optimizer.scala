package com.feynmanliang.optala

/** The results from a run of an iterative optimization algorithm along with performance diagnostic information.
  *
  * @param stateTrace sequence of states across iterations
  * @param numObjEval number of objective function evaluations until termination
  * @param numGradEval number of gradient evaluations until termination
  * @tparam T the type for the algorithm's states
  */
private[optala] case class OptimizationResult[T](
    stateTrace: List[T],
    numObjEval: Long,
    numGradEval: Long)

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
