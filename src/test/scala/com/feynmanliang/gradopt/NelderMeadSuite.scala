package com.feynmanliang.gradopt

import breeze.linalg._
import breeze.numerics._

import org.scalatest._

class NelderMeadSuite extends FunSpec {
  describe("Nedler-Mead") {
    describe("when applied to f(x,y) = x^2 + y^2") {
      val f: Vector[Double] => Double = x => x dot x
      var init = Simplex(
        List(
          DenseVector(-1D,.1D),
          DenseVector(-.1D,-3D),
          DenseVector(-2D,7D))
        .map(x => (x,f(x))))
      val nm = new NelderMead()

      println(
        nm.minimize(f, init)
        .drop(1000)
        .take(10)
        .map(_.points.map(_._2).min)
        .toList)

      println(
        nm.minimize(f, 2, 5)
        .drop(1000)
        .take(10)
        .map(s => {
          s.points.map(_._2).sum / (s.points.size * 1D)
        })
        .toList)
    }
  }
}

// vim: set ts=2 sw=2 et sts=2:
