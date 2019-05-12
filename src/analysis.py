import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

def save(d1, fname):
    d1.to_csv(fname, index=False)

def getUnique(d1):
    # Make Sure No Duplicate Models in DataFrame
    d1["N"].apply(lambda x : int(x))
    d1["C"] = [1/float(s.split("Lasso")[-1]) if len(s.split("Lasso")) > 1 else 0 for s in d1["Model Name"].values]
    d1["id"] = [d1.iloc[i]["Model Name"] + d1.iloc[i]["NGram"] + str(d1.iloc[i]["Stack"]) + str(d1.iloc[i]["Tfid"]) for i in range(len(d1))]
    d1.set_index("id")
    d1=d1.drop_duplicates(subset=['id'])
    d1 = d1.sort_values(by=['F1','Model Name', 'NGram', 'Stack'],ascending=False)
    return d1

def fixf1(f1, f2, fname="./all_lasso.csv"):
    d1 = pd.read_csv(f1)
    d2 = pd.read_csv(f2)
    d1 = getUnique(d1)
    d1 = d1[d1["Tfid"] == False]
    giant=d1.append(d2)
    giant.drop(columns=['id'])
    save(giant,fname)

#
# for the different regularization parameters, get
# the mean F1 values and regularization strength
#
def L1_Impact(d1):
    # For all of the runs
    # Group By Model Name - ModelName = Logistic Lasso + <Regularization Parameter>
    # Note sklearn regularization parameter is 'C', higher C is less regularization
    # We are graphing based on traditional alpha parameter which is saved in file as 1/C
    # Get the Mean F1 score for each model
    # Get L1 Regularization Impact 
    g=d1.groupby(['Model Name']).agg({
            'Model Name': [lambda x : ' '.join(x)],
            'C' :['mean'],
            'F1':['mean','min','max']})
    nameV = g['Model Name']['<lambda>'].values
    f1MeanV = g['F1']['mean'].values
    f1MinV = g['F1']['min'].values
    f1MaxV = g['F1']['max'].values
    CV = g['C']['mean'].values
    namveV =[s.split(" ")[-1] for s in nameV]
    combo=sorted([(name, ci, f1mean, f1min, f1max, ci) for name, ci, f1mean, f1min, f1max in zip(nameV, CV, f1MeanV, f1MinV, f1MaxV)], 
                 key=lambda x : x[1], reverse=False)
    N = len(combo)
    model = [None]* N
    param, f1Mean, f1Min, f1Max = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N) # 
    for i, c in enumerate(combo):
        model = c[0]
        param[i] = c[1]
        f1Mean[i] = c[2]
        f1Min[i] = c[3]
        f1Max[i] = c[4]
    return model, param, f1Mean, f1Min, f1Max

#
# Plot impact of regularization parameter on predicted F1 Score
#    
def PlotL1(f1, output=None):
    d1 = pd.read_csv(f1)
    dns = d1[d1["Stack"] == False]
    modelns, xns, yns, yns_min, yns_max = L1_Impact(dns)

    ds = d1[d1["Stack"] == True]
    models, xs, ys, ys_min, ys_max = L1_Impact(ds)
    
    plt.title('Logistic Lasso Regularization')
    plt.xlabel('L1 Regularization(larger more regularization)')
    plt.ylabel('F1 Score')

    plt.plot(xns[1:],yns[1:], marker='o', label='unstacked-mean')
    #plt.plot(xns[1:],yns_max[1:], marker='o', label='unstacked-max')
    #plt.plot(xns[1:],yns_min[1:], marker='o', label='unstacked-min')

    plt.plot(xs[1:],ys[1:], marker='o', label='stacked-mean')
    #plt.plot(xs[1:],ys_min[1:], marker='o', label='stacked-min')
    #plt.plot(xs[1:],ys_max[1:], marker='o', label='stacked-max')
    plt.legend()
    if output is not None:
        plt.savefig("{}.pdf".format(output), bbox_inches='tight')
    else:
        plt.show()

#
# Plot the Number of Elements in the Matricies for different NGrams with
# Stacking and UnStacking Documents
#   
def matrixSize(f1, output=None):
    d1 = pd.read_csv(f1)
    ng = sorted(d1["NGram"].unique(), key = lambda x : int(x.split(':')[0])*10 + int(x.split(':')[1]))
    ds = d1[d1["Stack"] == True]
    dns = d1[d1["Stack"] == False]
    label, nf_ns, nf_s = [], [], []
    skip = ['8:8', '12:12', '15:15'] # remove a couple of ngrams ot make it fit nicely on x axis
    for i in range(len(ng)):
        dxs = ds[ds['NGram'] == ng[i]]
        dxns = dns[dns['NGram'] == ng[i]]
        if len(dxs) == 0 or len(dxns) == 0: continue
        if ng[i] in skip : continue
        label.append(ng[i])
        nf_ns.append(dxns.sz.mean()/1e9)
        nf_s.append(dxs.sz.mean()/1e9)
    ind = np.arange(len(label)) 
    width = 0.15
    plt.title('Feature Matrix Sizes')
    plt.bar(ind-width/2,nf_s, width, label='stacked')
    plt.bar(ind+width/2,nf_ns, width, label='unstacked')
    plt.xticks(ind+width, label)
    plt.xlabel("ngrams")
    plt.ylabel("matrix elements (billions)")
    plt.legend()
    if output is not None:
        plt.savefig("{}.pdf".format(output), bbox_inches='tight')
    else:
        plt.show()

def sparse(f1, output=None):
    d1 = pd.read_csv(f1)
    ds = d1[d1["Stack"] == True]
    nfs = ds["NF"].values/1e6
    svs = ds["sparcity"].values
    f1s = ds["F1"].values
    #plt.scatter(sv, f1)
    plt.scatter(nfs, svs, marker='o', label='stacked')

    dns = d1[d1["Stack"] == False]
    dns = dns[dns["NGram"] != "1:1"]
    nfns = dns["NF"].values/1e6
    svns = dns["sparcity"].values
    f1ns = dns["F1"].values
    mx = dns["sparcity"].max()
    plt.scatter(nfns, svns, marker='x', label='unstacked')
    plt.legend()
    plt.title('Number of Features and Matrix Sparcity')
    plt.xlabel('Number of Features (Millions)')
    plt.ylabel('Matrix Sparcity')
    if output is not None:
        plt.savefig("{}.pdf".format(output), bbox_inches='tight')
    else:
        plt.show()

def naiveBayesSmoothing(f1, output=None):
    d1 = pd.read_csv(f1)
    spliton="MultiNB"
    d1["C"] = [float(s.split(spliton)[-1]) if len(s.split(spliton)) > 1 else 0 for s in d1["Model Name"].values]
    model, x, y, ymin, ymax= L1_Impact(d1)
    plt.plot(x,y, marker='o', label='mean f1')
    plt.plot(x,ymax, marker='o', label='max f1')
    plt.plot(x,ymin, marker='o', label='min f1')
    plt.xlabel('Laplace Smoothing Parameter')
    plt.ylabel('F1 score')
    plt.title('Naive Bayes Smothing Parameter')
    plt.legend()
    if output is not None:
        plt.savefig("{}.pdf".format(output), bbox_inches='tight')
    else:
        plt.show()

if __name__ == '__main__':
    # to get bar graph of matrix sizes
    # python ./analysis.py -i ../analysis/all_lasso.csv -r size -o [to save to file]
    #

    # to get bar graph of matrix sizes
    # python ./analysis.py -i ../analysis/all_lasso.csv -r size
    #

    # to get graph of features vs sparsity
    # python ./analysis.py -i ../analysis/all_lasso.csv -r sparse
    #

    # to get graph of impact of Smoothing Parmeter on Naive Bayes Models 
    # python analysis.py -i ../analysis/naive.csv  -r smooth


    parser = argparse.ArgumentParser(description='ML Spring 2019')
    parser.add_argument('-i','--input', nargs='+', default=None)
    parser.add_argument('-o','--output', default=None)
    parser.add_argument('-r','--run', help='l1 , size, sparse, smooth', default='l1')
    args = parser.parse_args()

    f1 = args.input[0]
    f2 = args.input[1] if len(args.input) >1 else None

    if args.run == 'l1':
        PlotL1(f1, args.output)
    elif args.run == 'size':
        matrixSize(f1, args.output)
    elif args.run == 'sparse':
        sparse(f1, args.output)
    elif args.run == 'smooth':
        naiveBayesSmoothing(f1, args.output)
