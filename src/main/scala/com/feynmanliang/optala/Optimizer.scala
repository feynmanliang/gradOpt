package com.feynmanliang.optala

/**
  * Detailed performance diagnostics from an iterative optimization run.
  * @param stateTrace sequence of internal algorithm states and objective values for each iteration
  * @param numObjEval the number of objective function evaluations
  * @param numGradEval the number of gradient evaluations
  * @tparam T the type of the interal algorithm state
  */
private[optala] case class PerfDiagnostics[T](
    stateTrace: List[T],
    numObjEval: Long, // make sure the Seq is evaluated before materializing
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
