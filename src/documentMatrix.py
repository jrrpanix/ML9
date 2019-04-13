import os
import csv
import glob
import numpy as np
import nltk
import string
import re

from numpy import genfromtxt
from nltk import *
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

def createDocumentMatrixWMetadata():
  ##Read FOMC History array
  FOMC_HISTORY = np.loadtxt('../text/history/RatesDecision.csv', dtype='str', delimiter=',')
  textArray =  np.empty((1,4),dtype='<U5')
  ##Path names to text files
  paths = ['../text/minutes/'],
           '../text/statements/'],
           '../text/speeches/']
  for i in paths:
    for files in glob.glob(i+"*.txt"):
    ##Regex patterns to clean the text 
      f = open(files)
      clean = re.sub("\\'",'',f.read()).strip()
      clean = re.sub("[^\x20-\x7E]", "",clean).strip()
      clean = re.sub("[0-9/-]+ to [0-9/-]+ percent","percenttarget ",clean)
      clean = re.sub("[0-9/-]+ percent","percenttarget ",clean)
      clean = re.sub("[0-9]+.[0-9]+ percent","dpercent",clean)
      clean = re.sub(r"[0-9]+","dd",clean)
      clean = re.sub("U.S.","US",clean).strip()
      clean = re.sub("p.m.","pm",clean).strip()
      clean = re.sub("a.m.","am",clean).strip()
      clean = re.sub("S&P","SP",clean).strip()
      clean = re.sub(r'(?<!\d)\.(?!\d)'," ",clean).strip()
      clean = re.sub(r"""
                   [,;@#?!&$"]+  # Accept one or more copies of punctuation
                   \ *           # plus zero or more copies of a space
                   """,
                   " ",          # and replace it with a single space
                   clean, flags=re.VERBOSE)
      clean = re.sub('--', ' ', clean).strip()  
      clean = re.sub("'",' ',clean).strip()
      clean = re.sub("- ","-",clean).strip()
      clean = re.sub('\(A\)', ' ', clean).strip()
      clean = re.sub('\(B\)', ' ', clean).strip()
      clean = re.sub('\(C\)', ' ', clean).strip()
      clean = re.sub('\(D\)', ' ', clean).strip()
      clean = re.sub('\(E\)', ' ', clean).strip()
      clean = re.sub('\(i\)', ' ', clean).strip()
      clean = re.sub('\(ii\)', ' ', clean).strip()
      clean = re.sub('\(iii\)', ' ', clean).strip()
      clean = re.sub('\(iv\)', ' ', clean).strip()
      clean = re.sub('/^\\:/',' ',clean).strip()
      clean=re.sub('\s+', ' ',clean).strip()
      ##Obtain Metadata from file name strings for numpy array
      ##Get document type
      doc_type = i.split("/")
      ##Get meeting date
      if(doc_type[2] == 'minutes'):
        meeting_date = files.split('/')
        meeting_date = meeting_date[3].split('_')[0]
        ##Get Fed action label
        x=np.where(FOMC_HISTORY == meeting_date)
        if (FOMC_HISTORY[x[0]][0][4] == 'unchg'):
          action = 'unchg'
        else:
          action = 'move'
      elif(doc_type[2] == 'statements'):
        meeting_date = files.split('/')
        meeting_date = meeting_date[3].split('_')[0].split('.')[0]
        #Get Fed action label
        x = np.where(FOMC_HISTORY == meeting_date)
        if (FOMC_HISTORY[x[0]][0][4] == 'unchg'):
          action = 'unchg'
        else:
          action = 'move'
      elif(doc_type[2] == 'speeches'):
        meeting_date = files.split('/')
        meeting_date = meeting_date[3].split('_')[0].split('.')[0]
        #Get Fed action label
        #print('speeches')
        #print(meeting_date)
        previous = ''
        for x in FOMC_HISTORY:
          if(x[0] > meeting_date):  
            meeting_date = previous
            break    
          previous = x[0]          
        x = np.where(FOMC_HISTORY == meeting_date)
        if (FOMC_HISTORY[x[0]][0][4] == 'unchg'):
          action = 'unchg'
        else:
          action = 'move'
      #print(action)
      textArray = np.append(textArray,[[action,meeting_date,doc_type[2],clean.strip().lower()]],axis=0)
  textArray = np.delete(textArray,0,0)
  numpy.save("/home/jjl359/ML/ML9/text/document-matrices/document_matrix.npy",textArray)   
  print("saved")

def main():
  createDocumentMatrixWMetadata()
  
  
if __name__ == '__main__':
  main()
  
  
  
  
