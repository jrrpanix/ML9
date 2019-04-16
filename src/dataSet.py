import os
import sys
import argparse
import numpy as np
import pandas as pd
import datetime
import bisect
import re
from clean import simple_clean
from sklearn.feature_extraction.text import CountVectorizer

#
# create a training set from speeches, statements, prior minutes
#

# convert a numpy.datetime64 to python datetime
def todt(date):
    if type(date) == datetime.datetime : return date
    ts = (date - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.datetime.utcfromtimestamp(ts)


def get_ge(datesV, date):
    # if datesV is np64
    #ib = bisect.bisect_right(datesV, np.datetime64(date))
    ib = bisect.bisect_right(datesV, date)
    return ib

def get_lt(datesV, date):
    # if datesV is np64
    #ib = bisect.bisect_right(datesV, np.datetime64(date))
    ib = bisect.bisect_left(datesV, date) - 1
    return ib


#
def getRatesDecisionDF(decisionFile):
    names = ["minutes_date","publish_date","before","after","decision","flag","change"]
    usecols = [0,1,2,3,4,5,6]
    dtypes={"minutes_date":'str',"publish_date":'str',"before":'float',"after":'float',
            "decision":'str',"flag":'float',"change":'float'}
    df = pd.read_csv(decisionFile, 
                     usecols=usecols,
                     header=None, 
                     names=names,
                     dtype=dtypes,
                     sep=",")
    df['minutes_date'] = pd.to_datetime(df['minutes_date'],format="%Y%m%d")
    df['publish_date'] = pd.to_datetime(df['publish_date'],format="%Y%m%d")
    return df

def getMinutesDates(mdir):
    minutes, publish, docs = [] ,[], []
    for files in sorted(os.listdir(mdir)):
        f, ext = os.path.splitext(files)
        if len(f.split("_")) != 4 : continue
        minutes.append(datetime.datetime.strptime(f.split("_")[0],"%Y%m%d"))
        publish.append(datetime.datetime.strptime(f.split("_")[-1],"%Y%m%d"))
        docs.append(files)
    return minutes, publish , docs


def getSpeeches(sdir):
    sdate, docs = [],[]
    for files in sorted(os.listdir(sdir)):
        f, ext = os.path.splitext(files)
        if len(f.split("_")) < 2 : continue
        sdate.append(datetime.datetime.strptime(f.split("_")[0],"%Y%m%d"))
        docs.append(files)
    return sdate, docs

def getStatements(sdir):
    sdate, docs = [],[]
    for files in sorted(os.listdir(sdir)):
        f, ext = os.path.splitext(files)
        sdate.append(datetime.datetime.strptime(f,"%Y%m%d"))
        docs.append(files)
    return sdate, docs

def getTextVector(ddir, docList):
    def getText(fullpath):
        try:
            f = open(fullpath)
            return f.read()
        except:
            print("bad file %s" % fullpath)
            return "x"
    return [getText(os.path.join(ddir, doc)) for doc in docList]

def dec2f(decision):
    if decision == "unchg" : return 0.0
    if decision == "raise" : return 1.0
    if decision == "raise" : return -1.0

class DataSet:

    def __init__(self, decisionFile, minutesDir, speechDir, statementDir):
        # rates decision df "../text/history/RatesDecision.csv")
        self.ratesdf = getRatesDecisionDF(decisionFile)

        # convert to a vector of datetime , pandas dates are np.datetime64, for consistency
        self.publishDates = [todt(d64) for d64 in self.ratesdf["publish_date"].values]
        self.decision = self.ratesdf["decision"].values
        self.decisionValue = [dec2f(d) for d in self.decision]
        self.flag = self.ratesdf["flag"].values

        # publish date of minutes and vector of all text in
        # fiels in directory "../text/minutes")
        self.minutesDates, self.minutesPublishDates, self.minutesDocs = getMinutesDates(minutesDir)
        self.minutesText = getTextVector(minutesDir, self.minutesDocs)
        
        
        # dates of speeches along with vector of all speechs text
        # "../text/speeches")
        self.speechDates, self.speechDocs = getSpeeches(speechDir)
        self.speechText = getTextVector(speechDir, self.speechDocs)

        # dates of statements along with vector of all statement texts
        self.statementDates, self.statementDocs = getStatements(args.statements)
        self.statementText = getTextVector(statementDir, self.statementDocs)


    def calcDataTuple(self):
        # merge response to data
        # this calculate the response and the greatest index for each of the texts
        # that can be used as input to the response.
        # a hpyer parameter can then be used to determine the lookback
        #

        # prediction is 
        # Action, No Action Model
        # P(RateDecision(n)=1|RateDecision(n-1)..RateDec(0), statemets < RateDecision, speeches < RateDecision)
        # P(RateDecision(n)=0|RateDecision(n-1)..RateDec(0), statemets < RateDecision, speeches < RateDecision)
        
        # prediction is raise, lower, unchg
        #
        # P(RateDecision(n)=raise|RateDecision(n-1)..RateDec(0), statemets < RateDecision, speeches < RateDecision)
        # P(RateDecision(n)=lower|RateDecision(n-1)..RateDec(0), statemets < RateDecision, speeches < RateDecision)
        # P(RateDecision(n)=unchg|RateDecision(n-1)..RateDec(0), statemets < RateDecision, speeches < RateDecision)

        dataTuple=[]
        # note pandas datetime stored as numpy.datetime64, f64
        for ix, (publishDate, flag,  decision) in enumerate(zip(self.publishDates, self.flag,  self.decisionValue)):
            
            # skip a couple 
            if ix < 2 : continue
            print(publishDate)
             
            # prior minutes
            minutes_ix = ix - 1

            # find latest speech date occurring before publish date
            speech_ix = get_lt(self.speechDates, publishDate)

            # find statement date 
            statement_ix = get_lt(self.statementDates, publishDate)
            #stdt = self.statementDates[statement_ix] if statement_ix > 0 else None
            #print(stdt, publishDate, statement_ix)
            dataTuple.append([(publishDate,flag, decision, ix),(minutes_ix, speech_ix, statement_ix)])
        return dataTuple

    # some other stuff for later use
    #
    # TF(word,text) = # occurrcnes/total words
    #
    # IDF(word) = log[ #texts/#texts where word occurrs]
    #
    # TD-IDF(word,text) = TD*IDF


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SocialNetworks CSV to HDF5')
    parser.add_argument('--decision', default="../text/history/RatesDecision.csv")
    parser.add_argument('--minutes', default="../text/minutes")
    parser.add_argument('--speeches', default="../text/speeches")
    parser.add_argument('--statements', default="../text/statements")
    args = parser.parse_args()

    dataset = DataSet(args.decision, args.minutes, args.speeches, args.statements)
    dataTuple = dataset.calcDataTuple()

    #
    #
    #vectorizer = CountVectorizer(stop_words="english", preprocessor=simple_clean)
    #training_features = vectorizer.fit_transform(statementText)
    #test_features = vectorizer.transform(minutesText)
    

