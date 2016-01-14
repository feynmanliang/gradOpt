package com.feynmanliang.optala

import breeze.linalg._
import breeze.numerics._

/** An interval bracketing a step size which contains the line search minima.
  *
  * Note that a starting point, search direction, and objective function are implicit i.e. fMid = f(x + mid*p) where
  * "f" is the objective function, "x" is the starting point, and "p" is the search direction.
  *
  * The BracketInterval must satisfy lb <= mid, mid <= ub, fLb >= fMid and fMid =< fUB.
  * @param lb lower bound on step size
  * @param mid middle point on step size
  * @param ub upper bound on step size
  * @param fLb objective value at step size lower bound
  * @param fMid objective value at step size mid point
  * @param fUb objective value at step size upper bound
  */
private[optala] case class BracketInterval(
    lb: Double,
    mid: Double,
    ub: Double,
    fLb: Double,
    fMid: Double,
    fUb: Double) {
  require(lb <= mid && mid <= ub, s"bracket did not satisfy $lb <= $mid <= $ub")
  def contains(x: Double): Boolean = lb <= x && ub >= x
  def size: Double = ub - lb
  def bracketsMin: Boolean = fLb >= fMid && fMid <= fUb
}

object BracketInterval {
  /** Factory method for instantiating `Solution`s */
  def apply(phi: Double => Double, lb: Double, mid: Double, ub: Double): BracketInterval =
    BracketInterval(lb, mid, ub, phi(lb), phi(mid), phi(ub))
}

object LineSearch {
  private val GOLD = 1.61803398875
  private val EPS_MIN = 1E-20

  /** Brackets a step size "alpha" such that for some step size within the bracket
    * the restriction of f to the ray f(x + alpha*p) is guaranteed to attain
    * a minimum.
    * @param f objective function
    * @param df gradient
    * @param x starting point
    * @param p line search direction
    * @param initialBracketSize initial guess for bracket size
    * @param maxBracketIters maximum number of bracketing iterations before failing
    * @return Some(BracketInterval) if success, None otherwise
    */
  private[optala] def bracket(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x: Vector[Double],
      p: Vector[Double],
      initialBracketSize: Double = 100D,
      maxBracketIters: Int = 5000): Option[BracketInterval] = {
    val (phi, dPhi) = restrictRay(f, df, x, p)
    val initBracket = BracketInterval(phi, 0, initialBracketSize, (1 + GOLD) * initialBracketSize)
    if (norm(dPhi(0)) == 0.0D) {
      Some(initBracket)
    } else {
      /** An infinite stream of successively expanding brackets. */
      def nextBracket(currBracket: BracketInterval): Stream[BracketInterval] = currBracket match {
        case BracketInterval(lb, mid, ub, fLb, fMid, fUb) =>
          val newLb = if (fMid < fLb) lb else lb + (lb - mid)
          val newUb = if (fMid < fUb) ub else ub + (ub - mid)
          currBracket #:: nextBracket(BracketInterval(phi, newLb, mid, newUb))
      }
      nextBracket(initBracket)
        .take(maxBracketIters)
        .find(_.bracketsMin)
    }
  }

  /** Performs a bracketing line search along phi(a) = f(x + a*p) to determine a step size satisfying Strong Wolfe
    * Conditions.
    * @param f objective function
    * @param df gradient
    * @param x initial point
    * @param p line search direction
    * @param c1 Wolfe condition C1 constant
    * @param c2 Wolfe condition C2 constant
    * @return Some(alpha) of the step size satisfying Strong Wolfe Conditions if success, None otherwise
    */
  def chooseStepSize(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x: Vector[Double],
      p: Vector[Double],
      c1: Double = 1E-4,
      c2: Double = 0.9): Option[Double] = LineSearch.bracket(f, df, x, p) match {
    case _ if norm(p.toDenseVector) < EPS_MIN => Some(0D) // degenerate ray direction
    case None => None // FAIL: unable to bracket a minimum
    case Some(bracket) =>
      val aMax = bracket.size
      val initAlphaBracketSize = EPS_MIN // initial size for bracketing of alpha satisfying Strong Wolfe Conditions

      val (phi, dPhi) = restrictRay(f, df, x, p)
      val phiZero = phi(0)
      val dPhiZero = dPhi(0)

      /** Finds a step length alpha while ensures that (aPrev, aCurr) contains a point satisfying the Strong Wolfe
        * Conditions at each iteration.
        *
        * @see {Algorithm 3.5, Nocedal}
        *
        * @param aPrev previous step size
        * @param phiPrev phi(aPrev)
        * @param aCurr current step size
        * @param firstIter flag indicating if this is the first iteration
        * @return Some(alpha) of the step size satisfying Strong Wolfe Conditions if success, None otherwise
        */
      def findAlpha(aPrev: Double, phiPrev: Double, aCurr: Double, firstIter: Boolean): Option[Double] = {
        if (aCurr > aMax) {
          Some(aMax)
        } else {
          val phiCurr = phi(aCurr)
          if (phiCurr > phiZero + c1 * aCurr * dPhiZero || (phiCurr >= phiPrev && !firstIter)) {
            zoom(aPrev, aCurr)
          } else {
            val dPhiCurr = dPhi(aCurr)
            if (math.abs(dPhiCurr) <= -1 * c2 * dPhiZero) {
              Some(aCurr)
            } else if (dPhiCurr >= 0) {
              zoom(aCurr, aPrev)
            } else {
              findAlpha(aCurr, phiCurr, (aCurr + aMax)*1.1 / 2D, firstIter = false)
            }
          }
        }
      }

      /** Generates alpha_j between alpha_{lo} and alpha_{hi}, and refines one of the two endpoints while ensuring
        * Wolfe conditions hold.
        *
        * @see {Algorithm 3.6, Nocedal}
        *
        * @param aLo upper bound on bracket interval
        * @param aHi lower bound on bracket interval
        * @return Some(alpha) of the step size satisfying Strong Wolfe Conditions if success, None otherwise
        */
      def zoom(aLo: Double, aHi: Double): Option[Double] = {
        assert(!aLo.isNaN && !aHi.isNaN)
        interpolate(aLo, aHi) match {
          case Some(aCurr) if aCurr >= aMax => Some(aMax)
          case Some(aCurr) if math.abs(aHi - aLo) > EPS_MIN =>
            val phiACurr = phi(aCurr)
            if (phiACurr > phiZero + c1 * aCurr * dPhiZero || phiACurr >= phi(aLo)) {
              zoom(aLo, aCurr)
            } else {
              val dPhiCurr = dPhi(aCurr)
              if (math.abs(dPhiCurr) <= -c2 * dPhiZero) {
                Some(aCurr)
              } else if (dPhiCurr * (aHi - aLo) >= 0) {
                zoom(aCurr, aLo)
              } else {
                zoom(aCurr, aHi)
              }
            }
          case _ => Some((aHi + aLo) / 2D)
        }
      }

      /** Finds the minimizer for the cubic interpolation within line search.
        * @param prev previous step size
        * @param curr current step size
        * @return
        */
      def interpolate(prev: Double, curr: Double): Option[Double] = {
        val d1 = dPhi(prev) + dPhi(curr) - 3D * (phi(prev) - phi(curr)) / (prev - curr)
        val d2 = signum(curr - prev) * sqrt(pow(d1, 2) - dPhi(prev) * dPhi(curr))
        val res = curr - (curr - prev) * (dPhi(curr) + d2 - d1) / (dPhi(curr) - dPhi(prev) + 2D * d2)

        if (!res.isNaN) Some(res) else None
      }
      findAlpha(0, phiZero, initAlphaBracketSize, firstIter = true)
  }

  /** Restricts a vector function `f` with derivative `df` along ray `f(x + alpha * p)` */
  private[optala] def restrictRay(
      f: Vector[Double] => Double,
      df: Vector[Double] => Vector[Double],
      x: Vector[Double],
      p: Vector[Double]): (Double => Double, Double => Double) = {
    (alpha => f(x + alpha * p), alpha => df(x + alpha * p) dot p)
  }


  /** Computes the exact step size alpha for minimizing a quadratic form induced by A along the ray x + alpha*p.
    * @param A matrix inducing quadratic form
    * @param df current gradient at x
    * @param x current point
    * @param p line search direction
    * @return step size exactly minimizing line search
    */
  def exactLineSearch(
      A: Matrix[Double],
      df: Vector[Double],
      x: Vector[Double],
      p: Vector[Double]): Option[Double] = {
    if (norm(p.toDenseVector) == 0D) Some(0D) // degenerate ray
    else {
      val num = -(df.t * p)
      val denom = p.t * (A * p) // assumes A is PSD
      denom match {
        case 0 => None
        case _ => Some(num / denom)
      }
    }
  }
}
