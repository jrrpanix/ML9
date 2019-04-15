import os
import csv
import glob
import numpy as np
import nltk
import string
import re
import datetime
import bisect

from numpy import genfromtxt
from nltk import *
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk import word_tokenize
from nltk.util import ngrams
from Decision import getDecisionDF

def cleanFile(files):
    f = open(files)
    try:
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
    except:
        print("Unable to clean file %s" % files)
        return None
    return clean

def getMeetingDate(files):
    files, ext = os.path.splitext(files)
    fV = files.split('/')
    if len(fV) < 4 :
        #print("Bad File %s" % files)
        return None
    dstr = fV[3].split('_')[0]
    if len(dstr) != 8 :
        #print("Bad Date in File %s" % files)
        return None
    return datetime.datetime.strptime(dstr,"%Y%m%d")

def getDecision(HIST, meeting_date):
    ix=HIST.index[HIST['minutes_date'] == meeting_date].tolist()
    if len(ix) == 0:
        #print("ix date out of range %s" % meeting_date)
        return None
    ix = ix[0]
    return HIST.iloc[ix]["decision"]

def getAction(HIST, meeting_date):
    if meeting_date is None : return None
    decision = getDecision(HIST, meeting_date)
    if decision is None : return None
    return 'unchg' if decision == 'unchg' else 'move'

def getDocType(i):
    return i.split("/")[2]

def getPreviousDate(HIST, doc_date):
    datesV = HIST["minutes_date"].values
    ib = bisect.bisect_left(datesV, np.datetime64(doc_date))
    if ib > 1 and ib < len(HIST):
        return HIST.iloc[ib-1]["minutes_date"]
    return None

def CreateMatrix(HIST, paths, outputFile):
    textArray =  np.empty((1,4),dtype='<U5')
    good, errors = [],[]
    for i in paths:
        doc_type = getDocType(i)
        for files in glob.glob(i+"*.txt"):
            clean=cleanFile(files)
            meeting_date = getMeetingDate(files)
            if doc_type == "speeches" :
                meeting_date = getPreviousDate(HIST, meeting_date)
            action = getAction(HIST, meeting_date)
            if clean is None or doc_type is None or meeting_date is None or action is None:
                errors.append("{}:{}".format(i,files))
                print("failure %s" % files)
            else:
                good.append((meeting_date,files))
                textArray = np.append(textArray,[[action,meeting_date,doc_type,clean.strip().lower()]],axis=0)
    textArray = np.delete(textArray,0,0)
    numpy.save(outputFile,textArray)   
    print("finished processing num_ok=%d fails=%d, output=%s" % (len(good),len(errors), outputFile))


if __name__=='__main__':
    
    HIST = getDecisionDF('../text/history/RatesDecision.csv')
    paths = ['../text/minutes/',
             '../text/statements/',
             '../text/speeches/']
    outputFile = "docmatrix.npy"
    CreateMatrix(HIST, paths, outputFile)

