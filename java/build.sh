#!/bin/sh

curl -L https://www.antlr.org/download/antlr-runtime-4.13.2.jar -o java/antlrsrc.jar

antlr4 -Dlanguage=Java java/Solidity.g4

fd . ./java/ -e java -X javac -d java/out {}

jar cfm java/antlr.jar java/manifest.txt -C java/out .
