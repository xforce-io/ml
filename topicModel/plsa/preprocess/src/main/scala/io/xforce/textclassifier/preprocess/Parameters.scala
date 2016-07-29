package io.xforce.textclassifier.preprocess

import java.io.{BufferedReader, File, FileReader, FileWriter}

import org.ahocorasick.trie.Trie

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Created by nali on 16/7/19.
  */
class Parameters(dirpath :String, outputpath :String = "") {
  private val tmp = createWordsBag_
  private var files_ :Array[File] = tmp._1
  private var wordsBag_ :Array[String] = tmp._2
  private var wordToIndex_ = buildWordToIndex_
  private var numDocs_ = files_.length
  private var numWords_ = wordsBag_.length
  private val accuDocAndWords_ = createAccuDocAndWords_
  private val accuDocs_ = new Array[Int](numDocs_)
  private val acTrie_ = buildACTrie_

  //build arrays_
  for (i <- 0 until files_.length) {
    processFile_(files_(i), i)
    println("index[%d] file[%s]".format(i, files_(i).getName))
  }

  println("numDocs[%d] numWords[%d]".format(numDocs_, numWords_))

  def numDocs = numDocs_
  def numWords = numWords_
  def accuDocAndWords = accuDocAndWords_
  def accuDocs = accuDocs_

  private def processFile_(file :File, idxFile :Int) = {
    for (line <- Source.fromFile(file).getLines()) {
      val iter = acTrie_.parseText(line).iterator()
      while (iter.hasNext) {
        val item = iter.next()
        val idx = wordToIndex_.get(item.getKeyword).get
        accuDocAndWords_(idxFile)(idx) += 1
      }
    }

    for (i <- 0 until numDocs_) {
      var accu = 0
      for (j <- 0 until numWords_) {
        accu += accuDocAndWords_(i)(j)
      }
      accuDocs_(i) = accu
    }
  }

  private def createWordsBag_ :(Array[File], Array[String]) = {
    val file = new File(ServiceConfig.globalConfig.filepathWordsBag)
    if (file.exists()) {
      val ret = (new ArrayBuffer[File](), new ArrayBuffer[String]())
      val fileReader = new BufferedReader(new FileReader(file))
      val numFiles = fileReader.readLine().toInt
      for (i <- 0 until numFiles) {
        ret._1.append(new File(fileReader.readLine()))
      }

      val numWords = fileReader.readLine().toInt
      for (i <- 0 until numWords) {
        ret._2.append(fileReader.readLine())
      }
      (ret._1.toArray, ret._2.toArray)
    } else {
      val result = MultiWordUnitExtractor.getMultiWordUnitFromDir(dirpath)
      val fileWriter = new FileWriter(file)
      fileWriter.write("%d\n".format(result._1.length))
      for (file <- result._1) {
        fileWriter.write(file.getAbsolutePath + '\n')
      }

      fileWriter.write("%d\n".format(result._2.length))
      for (str <- result._2) {
        fileWriter.write(str+'\n')
      }
      fileWriter.close()
      result
    }
  }

  private def buildWordToIndex_ :mutable.HashMap[String, Int] = {
    val result = new mutable.HashMap[String, Int]()
    for (i <- 0 until wordsBag_.length) {
      result.put(wordsBag_(i), i)
      println("index[%d] word[%s]".format(i, wordsBag_(i)))
    }
    result
  }

  private def createAccuDocAndWords_ :Array[Array[Int]] = {
    val result = new Array[Array[Int]](numDocs_)
    for (i <- 0 until numDocs_) {
      result(i) = new Array[Int](numWords_)
    }
    result
  }

  private def buildACTrie_ :Trie = {
    val builder = Trie.builder().removeOverlaps()
    for (word <- wordsBag_) {
      builder.addKeyword(word)
    }
    builder.build()
  }
}
