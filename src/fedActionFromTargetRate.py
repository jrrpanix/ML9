import os
import csv
import glob
import numpy as np
import pandas as pd
import nltk
import string
import re

from numpy import genfromtxt
from nltk import *
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

def createRateMoves(pathToCSV):
  dailyRates = pd.read_csv(pathToCSV,dtype=object)
  priorRate = -1
  actionDF = pd.DataFrame()
  for index,row in dailyRates.iterrows():
    if(priorRate != row['DFEDTAR'] and row['date'] != 20081216):
      chg = float(row['DFEDTAR']) - float(priorRate)
      actionDF = actionDF.append({"Date":row['date'], "Rate":row['DFEDTAR'], "MinutesReleaseDate":"","Chg":str(chg)},ignore_index=True)
      priorRate = row['DFEDTAR']

  actionDF = actionDF[['Date','MinutesReleaseDate','Rate','Chg']]
  print(actionDF.loc[actionDF["Date"] == "20010103","Rate"])

  print(actionDF.dtypes)

def main():
  path = '../text/history/dailyRateHistory.csv'
  createRateMoves(path)

if __name__ == '__main__':
  main()
  
