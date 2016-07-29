#!/usr/bin/python

import os, subprocess, sys

mod_name = "preprocess"

def Stop() :
  cmd = "ps aux | grep %s-assembly | grep java | awk '{ print $2 }' | xargs -i kill -9 {}" % mod_name
  try :
    subprocess.check_output(cmd, shell=True)
  except Exception, e :
    pass

def Shutup() :
  cmd = "ps aux | grep %s-assembly | grep java | awk '{ print $2 }' | xargs -i kill -9 {}" % mod_name
  try :
    subprocess.check_output(cmd, shell=True)
  except Exception, e :
    pass

def Start() :
  cmd = "nohup java -Xmx32768m -Xloggc:log/gc.log -jar %s-assembly-1.0.jar &" % mod_name
  os.system(cmd)

def Init() :
  cmd = "java -Xmx32768m -Xloggc:log/gc.log -jar %s-assembly-1.0.jar init" % (mod_name)
  os.system(cmd)

if __name__ == "__main__" :
  if len(sys.argv) != 2 or \
      (   sys.argv[1] != "stop" and \
          sys.argv[1] != "start" and \
          sys.argv[1] != "restart" and \
          sys.argv[1] != "shutup" and \
          sys.argv[1] != "init") :
    print("Usage: ./server [stop|start|restart|shutup|init]")
  elif sys.argv[1] == "stop" :
    Stop()
  elif sys.argv[1] == "start" :
    Start()
  elif sys.argv[1] == "shutup" :
    Shutup()
  elif sys.argv[1] == "restart":
    Stop(); Start()
  elif sys.argv[1] == "init" :
    Init()
