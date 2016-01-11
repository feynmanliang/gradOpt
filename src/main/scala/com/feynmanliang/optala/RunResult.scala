package com.feynmanliang.optala

/** The results from a run of an iterative optimization algorithm along with performance diagnostic information.
  * @tparam T type of an algorithm's internal state
  */
trait RunResult[T] {
  val bestSolution: Solution // best solution found in this algorithm run
  val stateTrace: Seq[T] // sequence of states at each iteration
  val numObjEval: Long // number of objective function evaluations until termination
  val numGradEval: Long // number of gradient evaluations until termination
}


