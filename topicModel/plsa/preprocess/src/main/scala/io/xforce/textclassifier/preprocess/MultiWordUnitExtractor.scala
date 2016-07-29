package io.xforce.textclassifier.preprocess

import java.io.{File, StringReader}

import edu.stanford.nlp.tagger.maxent.MaxentTagger
import org.ahocorasick.trie.{Emit, Trie}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.control.Breaks._

/**
  * Created by nali on 16/7/16.
  */

class MultiWordUnitExtractor {

  private val tagger_ = new MaxentTagger(ClassLoader.getSystemResource("english-bidirectional-distsim.tagger").getFile)
  private val acTrie_ = buildACTrie_

  def getMultiWordUnitFromDir(dirpath :String) :(Array[File], Array[String]) = {
    val dir = new File(dirpath)

    //get all words
    val allFileToWords = new ArrayBuffer[(File, ArrayBuffer[String])]()
    for (file <- dir.listFiles() if file.getAbsolutePath.endsWith(".txt")) {
      allFileToWords.append((file, getMultiWordUnitFromFile_(file.getAbsolutePath)))
    }

    //preprocess
    val dict = new mutable.HashMap[String, Int]()
    for (fileToWords <- allFileToWords) {
      for (word <- fileToWords._2) {
        val item = dict.get(word)
        dict.put(word, if (item.isEmpty) 1 else item.get + 1)
      }
    }

    val dictAfterFilter = new mutable.HashSet[String]()
    for (pair <- dict) {
      if (pair._2 >= ServiceConfig.globalConfig.thresholdFreqWords) {
        dictAfterFilter.add(pair._1)
      }
    }

    //make result
    val resultFiles = new ArrayBuffer[File]()
    allFileToWords.foreach(fileToWords => {
      breakable {
        fileToWords._2.foreach(word => {
          if (dict.contains(word)) {
            resultFiles.append(fileToWords._1)
            break
          }
        })
      }
    })
    (resultFiles.toArray, dictAfterFilter.toArray.map {word => word.substring(0, word.size - 1).replace("_", " ")})
  }

  protected def getMultiWordUnitFromFile_(filepath :String) :ArrayBuffer[String] = {
    val start = System.currentTimeMillis()

    val result = new ArrayBuffer[String]
    var content = ""
    val lines = Source.fromFile(filepath).getLines()
    for (lineIter <- lines; line = lineMapper_(lineIter.toString)) {
      if (lineFilter_(line)) {
        content = content.concat(line)
      } else {
        content = content.concat("\n")
      }
    }

    val paragraphs = content.split("\n\n")
    for (paragraph <- paragraphs if paragraphFilter_(paragraph)) {
      val sentences = paragraph.split(Array(',', '.'))
      if (paragraph.count(c => c==',' || c=='.') >= sentences.size) {
        for (sentence <- sentences.map(sentenceMapper_).filter(sentenceFilter_)) {
          result.append(getMultiWordUnitFromStr_(sentence): _*)
        }
      }
    }

    val stop = System.currentTimeMillis()

    println("process[%s] cost[%d] units[%d] mem[%d]".format(
        filepath,
        (stop-start)/1000,
        result.size,
        getMemUsed_()))
    result
  }

  protected def getMultiWordUnitFromStr_(str :String) :ArrayBuffer[String] = {
    val (tag, words) = getPosStr_(str)

    val result = new ArrayBuffer[String]()
    val emits = acTrie_.parseText(tag)
    val iter = emits.iterator()
    while (iter.hasNext) {
      val item = iter.next()
      var unit = ""
      if (emitsFilter_(item)) {
        var cntNonTrivial = 0
        for (i <- item.getStart to item.getEnd) {
          unit = unit.concat("%s_".format(words(i)))
          if (words(i).length < 3) {
            if (words(i).length == 1) {
              cntNonTrivial = 100
            } else {
              cntNonTrivial += 1
            }
          }
        }

        if (cntNonTrivial <= 1) {
          result.append(unit)
        }
      }
    }
    result
  }

  protected def getPosStr_(str :String) :(String, ArrayBuffer[String]) = {
    val start = System.currentTimeMillis()

    var resultTag = ""
    val resultWords = new ArrayBuffer[String]()

    val sentences = MaxentTagger.tokenizeText(new StringReader(str))
    val iter = sentences.iterator()
    while (iter.hasNext) {
      val tSentence = tagger_.tagSentence(iter.next())
      val iter2 = tSentence.iterator()
      while (iter2.hasNext) {
        val item = iter2.next()
        val tag = item.tag()
        tag(0) match {
          case 'J' => resultTag = resultTag.concat("a")
          case 'N' => resultTag = resultTag.concat("n")
          case 'I' => resultTag = resultTag.concat("p")
          case 'C' => {
            resultTag = resultTag.concat(if (tag == "CC") "c" else "u")
          }
          case _ => {
            resultTag = resultTag.concat("u")
          }
        }
        resultWords.append(item.word())
      }
    }

    val stop = System.currentTimeMillis()
    if (stop - start > 1000) {
      println("process_str_cost[%d] str[%s]".format(stop - start, str))
    }
    (resultTag, resultWords)
  }

  protected def buildACTrie_ = {
    Trie.builder().removeOverlaps().addKeyword("an").
        addKeyword("n").
        addKeyword("nn").
        addKeyword("aan").
        addKeyword("ann").
        addKeyword("nan").
        addKeyword("nnn").
        addKeyword("npn").build()
  }

  protected def lineMapper_(str :String) :String = {
    str.concat("\n").replaceAll("[\\p{Digit}]+", " ")
  }

  protected def lineFilter_(str :String) :Boolean = {
    str.size > 5 && str.find(c => c.isLetter).isDefined
  }

  protected def paragraphFilter_(str :String) :Boolean = {
    str.contains('.') && str.size > 200
  }

  protected def sentenceMapper_(str :String) :String = {
    str.trim.toLowerCase().replaceAll("[\\p{Punct}\\p{Space}]+", " ")
  }

  protected def sentenceFilter_(str :String) :Boolean = {
    if (str.length == 0 || !str.contains(" ")) {
      return false
    }

    //remove formats
    val words = str.split(" ")
    if (words.size > 5 && words.count(word => word.length <= 2) * 1.0 / words.size > 0.5) {
      return false
    }
    true
  }

  protected def emitsFilter_(emit :Emit): Boolean = {
    !(emit.getStart == emit.getEnd && emit.getKeyword.size < 3)
  }

  protected def getMemUsed_(): Long = {
    (Runtime.getRuntime().maxMemory() - Runtime.getRuntime().freeMemory())/1024/1024
  }
}

object MultiWordUnitExtractor {
  val multiWordUnitExtractor = new MultiWordUnitExtractor

  def getMultiWordUnitFromDir(dirpath :String) :(Array[File], Array[String]) = {
    multiWordUnitExtractor.getMultiWordUnitFromDir(dirpath)
  }
}