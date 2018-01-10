#!/usr/bin/env python
# -*- coding:utf-8 -*-

import subprocess, sys
from urllib import quote
import json, pdb, re, requests

reload(sys)
sys.setdefaultencoding('utf8')

kMinThreshold = 0.05

def parseResultFilepath(resultFilepath) :
    topicToWord = {}
    docToTopic = {}
    for line in open(resultFilepath, "r") :
        if line[-1] == '\n' :
            line = line[:-1]

        if len(line) > 2 :
            if line[:2] == "jk" :
                items = line.split("=")
                if items[2] not in topicToWord.keys() :
                    topicToWord[items[2]] = []

                if float(items[3]) > kMinThreshold :
                    topicToWord[items[2]] += [(items[1], float(items[3]))]
            elif line[:2] == "ki" :
                items = line.split("=")
                if items[2] not in docToTopic.keys() :
                    docToTopic[items[2]] = []
                
                if float(items[3]) > kMinThreshold :
                    docToTopic[items[2]] += [(items[1], float(items[3]))]
                
    for topic in topicToWord.keys() :            
        tmp = sorted(topicToWord[topic], key = lambda x : -x[1])
        topicToWord[topic] = tmp[:3] if len(tmp) > 3 else tmp

    for doc in docToTopic.keys() :            
        tmp = sorted(docToTopic[doc], key = lambda x : -x[1])
        docToTopic[doc] = tmp[:3] if len(tmp) > 3 else tmp
    
    return (topicToWord, docToTopic)

def parseIdxFilepath(filepath) :
    dicts = {}
    for line in open(filepath, "r") :
        if line[-1] == '\n' :
            line = line[:-1]

        pair = line.split("\t")    
        dicts[pair[0]] = pair[1]
    return dicts    

if __name__ == "__main__" :
    if len(sys.argv) != 4 :
        print("./%s resultFilepath wordFilepath docFilepath" % (sys.argv[0]))
        sys.exit(1)

    resultFilepath = sys.argv[1]    
    wordFilepath = sys.argv[2]
    docFilepath = sys.argv[3]

    topicToWord, docToTopic = parseResultFilepath(resultFilepath)
    wordDict = parseIdxFilepath(wordFilepath)
    docDict = parseIdxFilepath(docFilepath)
    for doc in docToTopic.keys() :
        output = "%s(" % docDict[doc]
        topics = docToTopic[doc]
        for pair in topics :
            output += "%f => %s" % (pair[1], pair[0])
            words = topicToWord[pair[0]]
            for wordItem in words :
                output += "/%f:%s" % (wordItem[1], wordDict[wordItem[0]])
            output += " || "    
        output += ")"    
        print(output)
