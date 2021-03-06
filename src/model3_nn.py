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
import pandas as pd
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from clean import simple_clean
from clean import complex_clean
from modelutils import modelutils

# example usage
# run 3 nn first with one hidden layer with 10 units , then one with 2 hidden layers then one with 3 hidden layers
#python ./model3_nn.py --data minutes statements --layers 10 10,10 10,10,10  --max_iter 100 --stack --ngram 15:15

# to install pytorch
# conda install pytorch torchvision -c soumith
def runModels(models, model_data_set, Nitr, pctTrain, ngram):
    results, prec,recall,f1, trainPos, testPos=[],[],[],[], np.zeros(Nitr), np.zeros(Nitr)
    for iter in range(Nitr):
        train_data, test_data = modelutils.splitTrainTest(model_data_set, pctTrain)
        training_features, test_features = modelutils.getFeatures(train_data, test_data, ngram)
        for i, m in enumerate(models):
            model=m[1]
            model.fit(training_features, train_data["ActionFlag"])
            y_pred = model.predict(test_features)
            acc = accuracy_score(test_data["ActionFlag"], y_pred)
            if iter == 0:
                results.append(np.zeros(Nitr))
                prec.append(np.zeros(Nitr))
                recall.append(np.zeros(Nitr))
                f1.append(np.zeros(Nitr))
            results[i][iter]=acc
            prec_recall = precision_recall_fscore_support(test_data['ActionFlag'].tolist(), y_pred, average='binary') 
            prec[i][iter] = prec_recall[0]
            recall[i][iter] = prec_recall[1]
            f1[i][iter] = prec_recall[2]
        trainPos[iter] = train_data["ActionFlag"].sum()/ len(train_data)
        testPos[iter] = test_data["ActionFlag"].sum()/ len(test_data)
    return results, trainPos, testPos, prec, recall, f1




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Spring 2019')
    parser.add_argument('--decision', default="../text/history/RatesDecision.csv")
    parser.add_argument('--minutes', default="../text/minutes")
    parser.add_argument('--speeches', default="../text/speeches")
    parser.add_argument('--statements', default="../text/statements")
    parser.add_argument('--pctTrain', default=0.75, type=float)
    parser.add_argument('--Niter', help="number of times to refit data", default=3, type=int)
    parser.add_argument('--cleanAlgo', default="complex")
    parser.add_argument('--ngram', nargs='+', default=['1,1'])
    parser.add_argument('--max_iter', help="max iterations for sklearn solver", default=250, type=int)
    parser.add_argument('--solver', help="solver for sklearn algo", default='liblinear')
    parser.add_argument('--data', nargs="+", default=["minutes", "speeches", "statements"])
    parser.add_argument('--stack', action='store_true', default=False)
    parser.add_argument('--layers', nargs='+', default=['10,10'])
    parser.add_argument('-o','--output', default='../text/data_for_graphs/model2_anagrams.csv')
    args = parser.parse_args()

    clean_algo = complex_clean if args.cleanAlgo == "complex" else simple_clean
    pctTrain, cleanA, Niter, ngram = args.pctTrain, args.cleanAlgo, args.Niter, args.ngram
    solver, max_iter, datasetlist = args.solver, args.max_iter, args.data

    if len(ngram[0].split(':')) > 1 :
        ngram = ngram[0]
        lb , ub = int(ngram.split(':')[0]), int(ngram.split(':')[1])
        ngrams = [(i,i) for i in range(lb, ub+1)]
    else :
        ngrams = [(int(x.split(",")[0]),int(x.split(",")[1])) for x in ngram]

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

    models=[]
    for nn in args.layers:
        hidden=tuple([int(x) for x in nn.split(",")])
        model_name = "nn_" + "_".join(x for x in nn.split(","))
        models.append((model_name,MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter)))
        print("model {}".format(model_name))

    outputDF = []
    print("Determining Fed Action from minutes")
    print("%-20s %5s %5s %10s %10s %5s %8s %7s %10s %10s %-27s %6s %6s %6s %6s %6s %5s" % 
          ("Model Name", "NGram", "Niter", "mean(acc)", "std(acc)","N","PctTrain", "clean", "start", "end", "Data Sets", "TrainP", "TestP", "Prec", "Recall", "F1", "Stack"))

    for ngram in ngrams:
        results, trainPos, testPos, prec, recall, f1 = runModels(models, data_set, Niter, pctTrain, ngram)
        ngramstr = str(ngram[0]) + ":" + str(ngram[1])
        for m, r, t, u, v in zip(models, results, prec, recall, f1):
            name, mu, s, precMu, recallMu, f1Mu = m[0], np.mean(r), np.std(r), np.mean(t), np.mean(u), np.mean(v)
            print("%-20s %5s %5s %10.4f %10.4f %5d %8.3f %7s %10s %10s %-27s %6.3f %6.3f  %6.3f %6.3f  %6.3f %5s" % 
                  (name, ngramstr, Niter, mu, s, N, pctTrain, cleanA, start, end, datasetlabel, np.mean(trainPos), np.mean(testPos), precMu, recallMu, f1Mu, stack))

            outputDF.append([name, ngramstr, Niter, mu, s, N, pctTrain, cleanA, start, end, datasetlabel, np.mean(trainPos), np.mean(testPos), precMu, recallMu, f1Mu, stack])
        print("")

    outputDF = pd.DataFrame(outputDF)
    outputDF.to_csv(args.output, index=False, header=["Model Name", "NGram", "Niter", "mean(acc)", "std(acc)","N","PctTrain", "clean", "start", "end", "Data Sets", "TrainP", "TestP", "Prec", "Recall", "F1", "Stack"])




