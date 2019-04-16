import os
import sys
import argparse
import numpy as np
import pandas as pd
import datetime
import bisect

#
# create a training set from speeches, statements, prior minutes
#

def get_ge(datesV, date):
    ib = bisect.bisect_right(datesV, np.datetime64(date))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SocialNetworks CSV to HDF5')
    parser.add_argument('--decision', default="../text/history/RatesDecision.csv")
    parser.add_argument('--minutes', default="../text/minutes")
    parser.add_argument('--speeches', default="../text/speeches")
    parser.add_argument('--statements', default="../text/statements")
    args = parser.parse_args()

    # rates decision df
    ddf = getRatesDecisionDF(args.decision)

    # fed minutes
    minutes, publish, docs = getMinutesDates(args.minutes)
    
    # fed speeches
    sdate, docs = getSpeeches(args.speeches)

    # statements
    stdate, docs = getStatements(args.statements)

    #
    publishDates = ddf["publish_date"].values

    # index of first rate decison after speech
    speech_ix = [get_ge(publishDates, s) for i,s in enumerate(sdate)]

    # index of first rate decision after statement
    state_ix = [get_ge(publishDates, s) for i,s in enumerate(stdate)]

    print(state_ix[0],state_ix[-1], speech_ix[0], speech_ix[-1] ,len(publishDates))

    # prediction is 
    # Action, No Action Model
    # P(RateDecision(n)=1|RateDecision(n-1)..RateDec(0), statemets < RateDecision, speeches < RateDecision)
    # P(RateDecision(n)=0|RateDecision(n-1)..RateDec(0), statemets < RateDecision, speeches < RateDecision)

    # Model 2
    # raise, lower, unchg
    #
    # P(RateDecision(n)=raise|RateDecision(n-1)..RateDec(0), statemets < RateDecision, speeches < RateDecision)
    # P(RateDecision(n)=lower|RateDecision(n-1)..RateDec(0), statemets < RateDecision, speeches < RateDecision)
    # P(RateDecision(n)=unchg|RateDecision(n-1)..RateDec(0), statemets < RateDecision, speeches < RateDecision)

    
