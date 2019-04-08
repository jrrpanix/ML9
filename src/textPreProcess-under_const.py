import os
import csv
import glob
import numpy as np
import nltk
from nltk import *
import string
import re

from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

translator = str.maketrans('', '', string.punctuation)

a = numpy.array([['label','date','text']])
path = '/home/jjl359/ML/ML9/text/minutes/'

for files in glob.glob(path+"*.txt"):
  f = open(files)
  clean = re.sub("\\'",'',f.read()).strip()
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
  clean = re.sub("'",'',clean).strip()
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
  a = np.append(a,[['label','date',clean.strip().lower()]],axis=0)
  
numpy.save("/home/jjl359/ML/ML9/text/document-matrices/document_matrix.npy",a)   

def main():


  
if __name__ == '__main__':
  main()
  
  
  
  
  
  
  
  
  
  
  
  
  
#np.append(a,[['apples','test','adfasfdasdf']],axis=0)


#corpusdir = '/home/jjl359/ML/ML9/text/minutes/'
#newcorpus = PlaintextCorpusReader(corpusdir, '.*')
#corpus  = nltk.Text(newcorpus.words())
#len(newcorpus.words()),[len(newcorpus.words(d)) for d in newcorpus.fileids()]

#corpus = []
#path = '/home/jjl359/ML/ML9/text/minutes/'
#for files in glob.glob(path+"*.txt"):
#  f = open(files)
#  corpus.append(f.read())  

#frequencies = Counter([])

#for text in corpus:
#    token = nltk.word_tokenize(text)
#    bigrams = ngrams(token, 2)
#    frequencies += Counter(bigrams)

#for files in glob.glob(path+"*.txt"):
#  f = open(files)
#  a = np.append(a,[['label','date',f.read().translate(translator)]],axis=0)
