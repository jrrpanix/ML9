import os
import glob
import os
import sys
import argparse
import numpy as np
import pandas as pd
import datetime
import datetime as dt
import bisect
import re
import csv
import statistics

def minutesWordCount(minutesDir):
  totalWords, totalDoc = 0, 0
  words = 0
  docVec = []
  for files in sorted(os.listdir(minutesDir)):
    with open(minutesDir+'/'+files, 'r') as f:
      totalDoc += 1
      for line in f:
        line = line.split()
        totalWords += len(line)
        words += len(line)
      docVec.append(words)
      words = 0
  return totalWords, totalDoc, docVec

def statementsWordCount(direc):
  totalWords, totalDoc = 0, 0
  words = 0
  docVec = []
  for files in sorted(os.listdir(direc)):
    with open(direc+'/'+files, 'r',encoding='utf-8',errors='ignore') as f:
      totalDoc += 1
      for line in f:
        line = line.split()
        totalWords += len(line)
        words += len(line)
      docVec.append(words)
      words = 0
  return totalWords, totalDoc, docVec

def speechesWordCount(direc):
  totalWords, totalDoc = 0, 0
  words = 0
  docVec = []
  for files in sorted(os.listdir(direc)):
    with open(direc+'/'+files, 'r',encoding='utf-8',errors='ignore') as f:
      totalDoc += 1
      for line in f:
        line = line.split()
        totalWords += len(line)
        words += len(line)
      docVec.append(words)
      words = 0
  return totalWords, totalDoc, docVec

def printMetaData(docType,numWords,numDocs,wordVec):
  print('Total Words in '+docType+ ': '+str(numWords) + '\nTotal Docs: '+str(numDocs) + '\nAvg Words Per Doc: ' + str(numWords/numDocs) + '\nStDev: ' + str(statistics.stdev(wordVec, numWords/numDocs))+ '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Spring 2019')
    parser.add_argument('--minutes', default="../text/minutes")
    parser.add_argument('--speeches', default="../text/speeches")
    parser.add_argument('--statements', default="../text/statements")
    args = parser.parse_args()
    
    numWords, numDocs, wordVec = minutesWordCount(args.minutes) 
    printMetaData('minutes',numWords,numDocs,wordVec)
    
    numWords, numDocs, wordVec = speechesWordCount(args.speeches) 
    printMetaData('speeches', numWords,numDocs,wordVec)
    
    numWords, numDocs, wordVec = statementsWordCount(args.statements)
    printMetaData('statements',numWords,numDocs,wordVec)
