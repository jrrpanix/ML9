import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


"""
Compare Plot fits accuracy, precision, recall and f1 vs ngram length
"""
def fitScore(df, fit, output):
    ll = df[df["Model Name"] == fit]
    x = ll.NGramInt
    f1 = ll.F1
    recall = ll.Recall
    precision = ll.Prec
    
    acc = ll["mean(acc)"].values
    plt.title("{} vs NGram Length".format(fit))
    plt.plot(x, f1, label='F1')
    plt.plot(x, recall, label='Recall')
    plt.plot(x, precision, label='Precision')
    plt.plot(x, acc, label='Accuracy')
    plt.xlabel('NGram Length')
    plt.ylabel('Pct')
    plt.legend()
    if output is not None:
        plt.savefig("{}.pdf".format(output), bbox_inches='tight')
    else:
        plt.show()


"""
Compare F1 Scores of the Fits vs NGram Length
"""
def compareFits(df, fits, output, range):
    range = int(range) if range is not None  else len(df)
    for f in fits:
        ll = df[df["Model Name"] == f]
        ll = ll[ll["NGramInt"] <= range]
        x = ll.NGramInt
        f1 = ll.F1
        plt.plot(x, f1, marker='o', label='{} F1'.format(f))
    plt.title('F1 Scores: ' + ','.join(f for f in fits))
    plt.xlabel('NGram Length')
    plt.ylabel('Pct')
    plt.legend()
    if output is not None:
        plt.savefig("{}.pdf".format(output), bbox_inches='tight')
    else:
        plt.show()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Spring 2019')
    parser.add_argument('-i','--input', default=None)
    parser.add_argument('-o','--output', default=None)
    parser.add_argument('-r','--range', default=None)
    parser.add_argument('-f','--fit', nargs='+', help= "'svm', 'logistic_lasso', 'Naive_Bayes', 'logistic'", default='logistic_lasso')
    args = parser.parse_args()
    

    df = pd.read_csv(args.input)
    ng = df.NGram.values
    z = [int(x.split(':')[0]) for x in ng]
    df["NGramInt"] = z

    fits = [f.replace("_B"," B") for f in args.fit]
    if len(fits) == 1:
        fitScore(df, fits[0], args.output)
    if len(fits) > 1:
        compareFits(df, fits, args.output, args.range)





