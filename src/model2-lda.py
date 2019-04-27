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
from sklearn.metrics import precision_recall_fscore_support
from sklearn import decomposition
from sklearn.decomposition import LatentDirichletAllocation

# this version allows for multiple fit methods (SVM, Logistic, Logistic Lasso)
# this method also computes the mean and std deviation of the accuracy by
# this version also allows for ngrams
# allows the text preprocessing algorithm to be selected
# running each model Niter Times

# python ./model2_0.py --ngram 1,1 2,2 3,3 1,3 2,3 --Niter 3 --pctTrain .8 --data minutes speeches statements

def print_top_words(model, feature_names, n_top_words):
    print()
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += "|".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
        print()
    print()

def runModels(model_data_set):
    docs = pd.concat(model_data_set)
    docs = docs["DocText"]

    vectorizer = CountVectorizer(min_df = 5,
      stop_words='english',
      ngram_range=(1,2),
      lowercase=True)        
    wordMatrix = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=4,learning_method="online",max_iter=50)
    lda.fit(wordMatrix)
    tf_feature_names = vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, 15)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SocialNetworks CSV to HDF5')
    parser.add_argument('--decision', default="../text/history/RatesDecision.csv")
    parser.add_argument('--minutes', default="../text/minutes")
    parser.add_argument('--speeches', default="../text/speeches")
    parser.add_argument('--statements', default="../text/statements")
    parser.add_argument('--pctTrain', default=0.75, type=float)
    parser.add_argument('--Niter', help="number of times to refit data", default=10, type=int)
    parser.add_argument('--cleanAlgo', default="complex")
    parser.add_argument('--ngram', nargs='+', default=['1,1'])
    parser.add_argument('--max_iter', help="max iterations for sklearn solver", default=25000, type=int)
    parser.add_argument('--solver', help="solver for sklearn algo", default='liblinear')
    parser.add_argument('--data', nargs="+", default=["minutes", "speeches", "statements"])
    parser.add_argument('--stack', action='store_true', default=False)
    args = parser.parse_args()

    ngrams = [(int(x.split(",")[0]),int(x.split(",")[1])) for x in args.ngram]
    clean_algo = complex_clean if args.cleanAlgo == "complex" else simple_clean
    pctTrain, cleanA, Niter, ngram = args.pctTrain, args.cleanAlgo, args.Niter, args.ngram
    solver, max_iter, datasetlist = args.solver, args.max_iter, args.data

    assert len(datasetlist) > 0, "no data sets specified"
    datasetlabel=":".join(d for d in datasetlist)

    # fed rate decison matrix
    df = modelutils.decisionDF(args.decision)

    # datsets
    data_set, N, start, end = [], 0, datetime.datetime.now(), datetime.datetime(1970, 1, 1)
    if "minutes" in datasetlist:
        data_set.append(modelutils.getMinutes(args.minutes, df, clean_algo))
        (N, start, end) = modelutils.getBounds(data_set[-1])
    if "speeches" in datasetlist:
        data_set.append(modelutils.getSpeeches(args.speeches, df, clean_algo))
        (N1, start1, end1) = modelutils.getBounds(data_set[-1])
        N , start, end = N + N1, min(start, start1), max(end, end1)
    if "statements" in datasetlist:
        data_set.append(modelutils.getStatements(args.statements, df, clean_algo))
        (N1, start1, end1) = modelutils.getBounds(data_set[-1])
        N , start, end = N + N1, min(start, start1), max(end, end1)
    assert N  > 0, "no data in data_set"
    start, end = start.strftime("%m/%d/%Y"), end.strftime("%m/%d/%Y")
    if args.stack:
        data_set = [modelutils.stackFeatures(data_set)]
    runModels(data_set)
