package io.xforce.textclassifier.preprocess

import java.io.{File, FileWriter}

/**
  * Created by nali on 16/7/19.
  */
class Plsa(dirpath :String, numTopics :Int) {
  private val parameters_ = new Parameters(dirpath)

  def dump(outputPath :String): Unit = {
    val accuDocs = new Array[Int](parameters_.numDocs)

    val fileWriter = new FileWriter(new File(outputPath))
    fileWriter.write("%d,%d,%d\n".format(numTopics, parameters_.numDocs, parameters_.numWords))
    for (i <- 0 until parameters_.numDocs) {
      fileWriter.write("%d".format(parameters_.accuDocAndWords(i)(0)))
      accuDocs(i) += parameters_.accuDocAndWords(i)(0)
      for (j <- 1 until parameters_.numWords) {
        fileWriter.write("\t%d".format(parameters_.accuDocAndWords(i)(j)))
        accuDocs(i) += parameters_.accuDocAndWords(i)(j)
      }
      fileWriter.write('\n')
    }
    fileWriter.close()
  }
}
