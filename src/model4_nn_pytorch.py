#
#
#
from __future__ import print_function
import sys
import os
import time
import datetime
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as udata 
from torch.autograd import Variable
import torch
from torchtext import data


from clean import simple_clean
from clean import complex_clean
from modelutils import modelutils

# to install pytorch
# conda install pytorch torchvision -c soumith

#
# Neural Net1 Fully Connected
#
class FCNet1(nn.Module):

    def __init__(self, N):
        super(FCNet1, self).__init__()

        # 2 hidden layer NN

        # layer1 x->
        self.fc1 = nn.Sequential(nn.Linear(N, N), nn.ReLU())

        #layer2 
        n2 = int(N/2)
        
        self.fc2 = nn.Sequential(nn.Linear(N, n2), nn.ReLU())

        #output could do three to predict if raise, lower or hold
        self.fc3 = nn.Sequential(nn.Linear(n2, 2), nn.Sigmoid())

    def forward(self, X):
        xo = self.fc1(X)
        xo = self.fc2(xo)
        return self.fc2(xo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Spring 2019')
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
        data_set = modelutils.stackFeatures(data_set)
        N = len(data_set)
        stack="True"
    else:
        stack="Flase"

    model_data_set = data_set
    train_data, test_data = modelutils.splitTrainTest(model_data_set, pctTrain)
    training_features, test_features = modelutils.getFeatures(train_data, test_data, ngrams[0])


    net = FCNet1(100)



