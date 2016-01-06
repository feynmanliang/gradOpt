package com.feynmanliang.optala.examples

import java.io.File

import breeze.linalg.{Matrix, csvwrite}

/**
  * Utility functions for example code and experiments.
  */
object ExampleUtils {
  def experimentWithResults(
      experimentName: String,
      resultFName: String)(results: => Matrix[Double]) = {
    println(s"=== Performing experiment: $experimentName ===")
    val resultsFile = new File(s"results/$resultFName")
    println(s"Writing results to: $resultsFile")
    csvwrite(resultsFile, results)
  }
}
