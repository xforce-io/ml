import AssemblyKeys._

assemblySettings

name := "preprocess"

version := "1.0"

scalaVersion := "2.10.6"

libraryDependencies ++= {
  Seq(
    "log4j"                % "log4j"             % "1.2.17",
    "org.slf4j"            % "slf4j-log4j12"     % "1.7.2",
    "com.google.guava"     % "guava"             % "18.0",
    "org.apache.zookeeper" % "zookeeper"         % "3.4.6",
    "org.scalatest"        % "scalatest_2.10"    % "2.2.2",
    "com.typesafe"         % "config"            % "1.2.1",
    "org.apache.thrift"    % "libthrift"         % "0.9.1",
    "org.apache.curator"   % "curator-framework" % "2.10.0",
    "edu.stanford.nlp" % "stanford-corenlp" % "3.5.2",
    "org.ahocorasick" % "ahocorasick" % "0.3.0"
  )
}
