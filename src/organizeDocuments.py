import os
import csv
import glob
import numpy as np
import string
import re
import pandas as pd

from numpy import genfromtxt
from collections import Counter

#FOMC_HISTORY[FOMC_HISTORY['a']==20181219]

class docOrganizer:
  def __init__(self):
    self.columnNames = ['meetingDate','documentDate','documentType','meetingMonth','actionFlag','identifier','doctext']
    self.textArray = pd.DataFrame(columns=self.columnNames)
    self.paths = ['../text/minutes/',
           '../text/statements/',
           '../text/speeches/']
    self.FOMC_HISTORY = pd.read_csv('../text/history/RatesDecision.csv', delimiter=',' ,names =['MeetingDate','MinutesRelease','PreviousTarget','PostTarget','Direction','ActionFlag','Amount'], parse_dates=['MeetingDate','MinutesRelease'])
    self.FOMC_HISTORY['ActionFlag'] = self.FOMC_HISTORY['ActionFlag'].astype('int')
  def createDocMatrix(self):   
    counter = 0
    for i in self.paths:
      for files in glob.glob(i+"*.txt"):
        f = open(files,encoding="utf-8",errors="ignore")
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
        doc_type = i.split("/")
        identifier = ''
        ##Get meeting date
        if(doc_type[2] == 'minutes'):
          meeting_date = files.split('/')
          document_date = files.split('/')
          meeting_date = meeting_date[3].split('_')[0]
          ##Get Fed action label
          document_date = document_date[3].split('_')[3].split('.')[0]
          x=self.FOMC_HISTORY[self.FOMC_HISTORY['MeetingDate'] == pd.to_datetime(meeting_date,format='%Y%m%d')].iloc[0]['Direction']
          identifier = 'FOMC'
          if (x == 'unchg'):
            action = 0
          else:
            action = 1
        elif(doc_type[2] == 'statements'):
          meeting_date = files.split('/')
          meeting_date = meeting_date[3].split('_')[0].split('.')[0]
          document_date = meeting_date
          #Get Fed action label
          x=self.FOMC_HISTORY[self.FOMC_HISTORY['MeetingDate'] == pd.to_datetime(meeting_date,format='%Y%m%d')].iloc[0]['Direction']
          identifier = 'FOMC'
          if (x == 'unchg'):
            action = 0
          else:
            action = 1
        elif(doc_type[2] == 'speeches'):
          meeting_date = files.split('/')
          identifier = files.split('/')
          identifier = identifier[3].split('_')[1].split('.')[0]
          meeting_date = meeting_date[3].split('_')[0].split('.')[0]
          document_date = meeting_date
          previous = ''
          for index,row in self.FOMC_HISTORY.iterrows():
            if(row['MeetingDate'] > pd.to_datetime(meeting_date,format='%Y%m%d')):  
              meeting_date = row['MeetingDate']
              break    
            previous = row['MeetingDate']          
          x=self.FOMC_HISTORY[self.FOMC_HISTORY['MeetingDate'] == pd.to_datetime(meeting_date,format='%Y%m%d')].iloc[0]['Direction']
          if (x == 'unchg'):
            action = 0
          else:
            action = 1
        counter = counter + 1
        self.textArray.loc[counter] = [meeting_date,document_date,doc_type[2],1,action,identifier,clean]
    return(self.textArray)
    
def main(): 
  do = docOrganizer()
  docMatrix = do.createDocMatrix()
    
if __name__ == '__main__':
  main()
