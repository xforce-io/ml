package io.xforce.textclassifier.preprocess

import java.io.File

import com.typesafe.config.{Config, ConfigFactory}

/**
  * Created by nali on 16/5/6.
  */
class GlobalConfig(config :Config) {
  val dirTexts = config.getString("dirTexts")
  val numTopics = config.getInt("numTopics")
  val outputFilepath = config.getString("outputFilepath")
  val thresholdFreqWords = config.getInt("thresholdFreqWords")
  val filepathWordsBag = config.getString("filepathWordsBag")
}

object ServiceConfig {
  protected val staticConfig_ = ConfigFactory.parseFile(new File("conf/preprocess.conf"))

  val globalConfig = new GlobalConfig(staticConfig_.getConfig("global"))
}
