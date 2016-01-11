package com.feynmanliang.optala

/** Wraps a function to count the number of times it is evaluated.
  *
  * @param f underlying function being counted
  * @tparam T input type of `f`
  * @tparam U return type of `f`
  */
private[optala] class FunctionWithCounter[-T, +U](f: T => U) extends Function[T, U] {
  var numCalls: Int = 0 // number of times `f` is called

  override def apply(t: T): U = {
    numCalls += 1
    f(t)
  }
}

