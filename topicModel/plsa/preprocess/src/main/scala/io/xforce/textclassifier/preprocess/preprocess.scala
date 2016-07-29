package io.xforce.textclassifier.preprocess

/**
  * Created by nali on 16/7/20.
  */
object Preprocess {
  def main(args :Array[String]): Unit = {
    val plsa = new Plsa(ServiceConfig.globalConfig.dirTexts, ServiceConfig.globalConfig.numTopics)
    plsa.dump(ServiceConfig.globalConfig.outputFilepath)
  }
}
