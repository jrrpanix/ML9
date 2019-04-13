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

def statementDate(elem):
  return elem[0]

def createRateMoves(pathToStatements,pathToMinutes,pathToCSV): 
  actionDF = pd.DataFrame()
  targetRateHistDF = pd.DataFrame()
  dailyRates = pd.read_csv(pathToCSV,dtype=object)
  priorRate = 0
  actionFlag = 0
  previousDayValue = 0
  direction = 'unchg'
  for index,row in dailyRates.iterrows():
    if(row['date'] > '20170101' and index < (len(dailyRates)-1)):
      row['DFEDTAR'] = dailyRates.iloc[index+1,1]
      #row['DFEDTAR'] = dailyRates[dailyRates['DFEDTAR']][index+1]    
    
    chg = float(row['DFEDTAR']) - float(priorRate) 
    if(chg>0):
      direction='raise'
      actionFlag = 1
    elif(chg<0):
      direction='lower'
      actionFlag=1
    else:
      direction='unchg'
      actionFlag=0
  
    targetRateHistDF = targetRateHistDF.append({"Date":row['date'],"MinutesRelease":"","PriorRate": priorRate,"Rate":row['DFEDTAR'],"Direction":direction,"ActionFlag":int(actionFlag),"Change":chg},ignore_index=True)
    priorRate = row['DFEDTAR']
  for file in list(glob.glob(pathToStatements+'*.txt')):
    actionDF = actionDF.append({"Date":str(file).split('/')[3].split('.')[0]},ignore_index=True)

  targetRateHistDF = targetRateHistDF[['Date','MinutesRelease','PriorRate','Rate','Direction','ActionFlag','Change']]
  
  #print(actionDF.loc[actionDF["Date"] == "20010103","Rate"])
  actionDF = actionDF.sort_values(by=['Date'])
  actionDF.index = pd.RangeIndex(len(actionDF.index)) 
  targetRateHistDF = targetRateHistDF[targetRateHistDF['Date'].isin(actionDF['Date'].tolist())]
  targetRateHistDF.index = pd.RangeIndex(len(targetRateHistDF.index))
#  print(targetRateHistDF)
  
  dateArray = []
  for file in list(glob.glob(pathToMinutes+'*.txt')):
    fileString = str(file).split('/')[3].split('.')[0].split('_')
    dateArray.append([fileString[0],fileString[3]])
  dateArray.sort(key=statementDate,reverse=True)
#  print(dateArray)
  for i in range(len(targetRateHistDF)):
    meetingDate = targetRateHistDF.iloc[i,0]
    for j in range(len(dateArray)):
      if(meetingDate>dateArray[j][0]):
        targetRateHistDF.iloc[i,1] = dateArray[j-1][1]
        break
  targetRateHistDF.iloc[0,1] = '20000323'
  # print(targetRateHistDF)
  targetRateHistDF.to_csv('../text/history/RatesDecision.csv',header=False, index=False, sep=',')
def main():
  path = '../text/history/dailyRateHistory.csv'
  pathTwo = '../text/statements/'
  pathThree = '../text/minutes/'
  createRateMoves(pathTwo,pathThree,path)
  
if __name__ == '__main__':
  main()
  
