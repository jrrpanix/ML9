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
from modelutils import modelutils

# this version allows for multiple fit methods (SVM, Logistic, Logistic Lasso)
# this method also computes the mean and std deviation of the accuracy by
# this version also allows for ngrams
# allows the text preprocessing algorithm to be selected
# running each model Niter Times

# python ./model2_0.py --ngram 1,1 2,2 3,3 1,3 2,3 --Niter 50 --pctTrain .8


def runModels(models, model_data_set, Nitr, pctTrain, ngram):
    def pctPos(tdata):
        tl, tpos = len(tdata), tdata['sentiment'].sum()
        print(tl, tpos, tpos/tl)

    results=[]
    for iter in range(Nitr):
        train_data, test_data = modelutils.splitTrainTest(model_data_set, pctTrain)
        training_features, test_features = modelutils.getFeatures(train_data, test_data, ngram)
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
    df = modelutils.decisionDF(args.decision)
    minutes, publish = modelutils.getMinutes(args.minutes, df, clean_algo)
    statements = modelutils.getStatements(args.statements, df, clean_algo)

    # Train on current minutes
    solver='liblinear'
    max_iter=20000 
    models=[("svm",LinearSVC(max_iter=max_iter)),
            ("logistic",LogisticRegression(solver=solver,max_iter=max_iter)),
            ("logistic_lasso",LogisticRegression(penalty='l1',solver=solver,max_iter=max_iter)),
            ("Naive Bayes",MultinomialNB())]

    print("Determining Fed Action from minutes")
    print("%-20s %5s %5s %10s %10s %5s %8s %6s %10s %10s" % ("Model Name", "NGram", "Niter", "mean(acc)", "std(acc)","N","PctTrain", "clean", "start", "end"))
    N = len(minutes) + len(statements)
    for ngram in ngrams:
        results= runModels(models, [minutes, statements], Niter, pctTrain, ngram)
        start, end = publish[0].strftime("%m/%d/%Y"), publish[-1].strftime("%m/%d/%Y")
        ngramstr = str(ngram[0]) + ":" + str(ngram[1])
        for m, r in zip(models, results):
            name, mu, s = m[0], np.mean(r), np.std(r) 
            print("%-20s %5s %5s %10.4f %10.4f %5d %8.3f %6s %10s %10s" % (name, ngramstr, Niter, mu, s, N, pctTrain, cleanA, start, end))
        print("")
            



