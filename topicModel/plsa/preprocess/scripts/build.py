import subprocess

mod_name = "preprocess"

def CALL(cmd, ignore_res=False) :
  try:
    retcode = subprocess.check_output(cmd, shell=True)
  except subprocess.CalledProcessError, e :
    if ignore_res==True : return None
    log = "not expected : cmd[%s] output[%s] retcode[%s]" % (cmd, e.output, e.returncode)
    raise AssertionError(log)
  return retcode[:-1]

CALL("sbt assembly")
CALL("mkdir -p build && cd build && mkdir -p log bin scripts data && cd -", True)
CALL("cp -r conf/ build/conf")
CALL("cp scripts/server.py build/scripts/server.py")
CALL("cp target/scala-2.10/%s-assembly-1.0.jar build/%s-assembly-1.0.jar" % (mod_name, mod_name))
CALL("mv build %s && tar czvf %s.tar.gz %s" % (mod_name, mod_name, mod_name))
CALL("mv %s build" % (mod_name))
