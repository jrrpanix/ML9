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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import csv

# more scalable version of model0
# this version allows for multiple fit methods (SVM, Logistic, Logistic Lasso)
# this method also computes the mean and std deviation of the accuracy by
# allows the text preprocessing algorithm to be selected
# running each model Niter Times

# python ./model1.py --Niter 100 --cleanAlgo complex
# python ./model1.py --Niter 100 --cleanAlgo simple

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

def getMinutes(minutesDir, decisionDF, clean_algo):
    minutes, publish , data = [], [], []
    for files in sorted(os.listdir(minutesDir)):
        f, ext = os.path.splitext(files)
        minutes.append(datetime.datetime.strptime(f.split("_")[0],"%Y%m%d"))
        publish.append(datetime.datetime.strptime(f.split("_")[-1],"%Y%m%d"))
        try:
            text = clean_algo(open(os.path.join(minutesDir, files)).read().strip())
            dec = df[df["publish_date"] == publish[-1]].iloc[0]["flag"]
            data.append([text, dec])
        except Exception as e:
            print("exception reading file %s" % files)
            print(e)
            quit()
    return data, publish

def getStatements(statementsDir, decisionDF, clean_algo):
    statements, data = [], []
    for files in sorted(os.listdir(statementsDir)):
      f, ext = os.path.splitext(files)
      statements.append(datetime.datetime.strptime(f.split(".")[0],'%Y%m%d'))
      try:
        text = clean_algo(open(os.path.join(statementsDir,files),encoding='utf-8',errors='ignore').read().strip())
        dec = df[df["publish_date"] == publish[-1]].iloc[0]["flag"]
        data.append([text,dec])
      except Exception as e:
        print("exception reading file %s" % files)
        print(e)
        quit()
    return data

def splitTrainTest(data, data_statements, trainPct):
    np.random.shuffle(data)
    Ntrain = int(len(data)*args.pctTrain)
    train_data = pd.DataFrame(data[0:Ntrain],columns=['text', 'sentiment'])
    train_data = train_data.append(pd.DataFrame(data_statements,columns=['text','sentiment']))
    test_data = pd.DataFrame(data[Ntrain:], columns=['text', 'sentiment'])
    return train_data, test_data

def getFeatures(train_data, test_data, ngram):
    vectorizer = CountVectorizer(stop_words="english",preprocessor=None, ngram_range=ngram)
    #vectorizer = TfidfVectorizer(stop_words="english",preprocessor=None,ngram_range=ngram)
    training_features = vectorizer.fit_transform(train_data["text"])                                 
    test_features = vectorizer.transform(test_data["text"])
    return training_features, test_features

def runModels(models, data, data_statements, Nitr, pctTrain, ngram):
    results=[]
    for iter in range(Nitr):
        train_data, test_data = splitTrainTest(data, data_statements, pctTrain)
        training_features, test_features = getFeatures(train_data, test_data, ngram)
        for i, m in enumerate(models):
            model=m[1]
            model.fit(training_features, train_data["sentiment"])
            y_pred = model.predict(test_features)
            acc = accuracy_score(test_data["sentiment"], y_pred)
            if iter == 0:
                results.append(np.zeros(Nitr))
            results[i][iter]=acc
    return results
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SocialNetworks CSV to HDF5')
    parser.add_argument('--decision', default="../text/history/RatesDecision.csv")
    parser.add_argument('--minutes', default="../text/minutes")
    parser.add_argument('--speeches', default="../text/speeches")
    parser.add_argument('--statements', default="../text/statements")
    parser.add_argument('--pctTrain', default=0.75, type=float)
    parser.add_argument('--Niter', default=10, type=int)
    parser.add_argument('--cleanAlgo', default="complex")
    parser.add_argument('--ngram', nargs='+', default=['1,1'])
    args = parser.parse_args()

    ngrams = [(int(x.split(",")[0]),int(x.split(",")[1])) for x in args.ngram]

    clean_algo = complex_clean if args.cleanAlgo == "complex" else simple_clean
    pctTrain, cleanA, Niter, ngram = args.pctTrain, args.cleanAlgo, args.Niter, args.ngram
    df = decisionDF(args.decision)
    data, publish = getMinutes(args.minutes, df, clean_algo)
    data_statements = getStatements(args.statements, df, clean_algo)
    N = len(data)
    # Train on current minutes
    models=[("svm",LinearSVC()),
            ("logistic",LogisticRegression(solver='liblinear')),
            ("logistic_lasso",LogisticRegression(penalty='l1',solver='liblinear')),
            ("logistic_l2",LogisticRegression(penalty='l2',solver='liblinear')),
            ("Naive Bayes",MultinomialNB())]

    print("Determining Fed Action from minutes")
    print("%-20s %5s %5s %10s %10s %5s %8s %6s %10s %10s" % ("Model Name", "NGram", "Niter", "mean(acc)", "std(acc)","N","PctTrain", "clean", "start", "end"))
    N = N + int(len(data_statements))
    for ngram in ngrams:
        results= runModels(models, data, data_statements, Niter, pctTrain, ngram)
        pctTrain = (int(len(data)*0.75) + int(len(data_statements))) / (int(len(data)) + int(len(data_statements)))
        start, end = publish[0].strftime("%m/%d/%Y"), publish[-1].strftime("%m/%d/%Y")
        ngramstr = str(ngram[0]) + ":" + str(ngram[1])
        for m, r in zip(models, results):
            name, mu, s = m[0], np.mean(r), np.std(r) 
            print("%-20s %5s %5s %10.4f %10.4f %5d %8.3f %6s %10s %10s" % (name, ngramstr, Niter, mu, s, N, pctTrain, cleanA, start, end))
        print("")
            



