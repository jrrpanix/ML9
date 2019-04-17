import os
import sys
import argparse
import numpy as np
import pandas as pd
import datetime
import bisect
import re
from clean import simple_clean
from clean import complex_clean
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def decisionDF(decisionFile):
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

def getMinutes(minutesDir, decisionDF):
    minutes, publish , data = [], [], []
    for files in os.listdir(minutesDir):
        f, ext = os.path.splitext(files)
        minutes.append(datetime.datetime.strptime(f.split("_")[0],"%Y%m%d"))
        publish.append(datetime.datetime.strptime(f.split("_")[-1],"%Y%m%d"))
        try:
            text = complex_clean(open(os.path.join(minutesDir, files)).read().strip())
            dec = df[df["publish_date"] == publish[-1]].iloc[0]["flag"]
            data.append([text, dec])
        except Exception as e:
            print("exception reading file %s" % files)
            print(e)
            quit()
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SocialNetworks CSV to HDF5')
    parser.add_argument('--decision', default="../text/history/RatesDecision.csv")
    parser.add_argument('--minutes', default="../text/minutes")
    parser.add_argument('--speeches', default="../text/speeches")
    parser.add_argument('--statements', default="../text/statements")
    parser.add_argument('--pctTrain', default=0.75, type=float)
    args = parser.parse_args()

    df = decisionDF(args.decision)
    data = getMinutes(args.minutes, df)
    np.random.shuffle(data)

    #
    # Train on current minutes
    # so the current meeting minutes are being used to predict if there's an action
    # obvious not legit because we need prior months more a proof of concept
    Ntrain = int(len(data)*args.pctTrain)
    train_data = pd.DataFrame(data[0:Ntrain],columns=['text', 'sentiment'])
    test_data = pd.DataFrame(data[Ntrain:], columns=['text', 'sentiment'])

    vectorizer = CountVectorizer(stop_words="english",preprocessor=None)
    training_features = vectorizer.fit_transform(train_data["text"])                                 
    test_features = vectorizer.transform(test_data["text"])

    model = LinearSVC()
    model.fit(training_features, train_data["sentiment"])
    y_pred = model.predict(test_features)
    acc = accuracy_score(test_data["sentiment"], y_pred)


    print("accuracy of model 0 %f" % acc)



