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
def ParameterImpact(d1):
    # For all of the runs
    # Group By Model Name - ModelName 

    # Logistic Lasso + <Regularization Parameter>
    # For Logistic Lasso the Inverse Regularization Parameter is 'C' in sklearn
    
    # Naive Bayes model names are MultiNB+<smoothing parameter>
    # For Naive Bayse the Laplace Smoothing Parameter is alpha in sklearn

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


def showOrSave(output=None):
    if output is not None:
        plt.savefig("{}.pdf".format(output), bbox_inches='tight')
    else:
        plt.show()
    
#
# Plot impact of regularization parameter on predicted F1 Score
#    
def PlotL1(f1, output=None, showMinMax=False):
    d1 = pd.read_csv(f1)
    dns = d1[d1["Stack"] == False]
    modelns, xns, yns, yns_min, yns_max = ParameterImpact(dns)

    ds = d1[d1["Stack"] == True]
    models, xs, ys, ys_min, ys_max = ParameterImpact(ds)
    
    plt.title('Logistic Lasso Regularization')
    plt.xlabel('L1 Regularization(larger more regularization)')
    plt.ylabel('F1 Score')

    plt.plot(xns[1:],yns[1:], marker='o', label='unstacked-mean F1 score')
    plt.plot(xs[1:],ys[1:], marker='o', label='stacked-mean F1 score')

    if showMinMax :
        plt.plot(xns[1:],yns_max[1:], marker='o', label='unstacked-max F1')
        plt.plot(xns[1:],yns_min[1:], marker='o', label='unstacked-min F1')
        plt.plot(xs[1:],ys_min[1:], marker='o', label='stacked-min F1')
        plt.plot(xs[1:],ys_max[1:], marker='o', label='stacked-max F1')

    plt.legend()
    showOrSave(output)

#
# Plot the Number of Elements in the Matricies for different NGrams with
# Stacking and UnStacking Documents
#   
def matrixSize(f1, output=None, f2=None):
    d1 = pd.read_csv(f1)
    if f2 is not None:
        d2 = pd.read_csv(f2)
        d1 = d1.append(d2)

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
    showOrSave(output)

#
# Plot the Number of Elements in the Matricies for different NGrams with
# Stacking and UnStacking Documents
#   
def matrixSparse(f1, output=None, f2=None):
    d1 = pd.read_csv(f1)
    if f2 is not None:
        d2 = pd.read_csv(f2)
        d1 = d1.append(d2)
    sparse_min = d1.sparcity.min()
    ng = sorted(d1["NGram"].unique(), key = lambda x : int(x.split(':')[0])*10 + int(x.split(':')[1]))
    ds = d1[d1["Stack"] == True]
    dns = d1[d1["Stack"] == False]
    label, nf_ns, nf_s = [], [], []
    skip = ['8:8', '12:12', '15:15'] # remove a couple of ngrams ot make it fit nicely on x axis
    skip = []
    for i in range(len(ng)):
        dxs = ds[ds['NGram'] == ng[i]]
        dxns = dns[dns['NGram'] == ng[i]]
        if len(dxs) == 0 or len(dxns) == 0: continue
        if ng[i] in skip : continue
        label.append(ng[i])
        nf_ns.append(dxns.sparcity.mean())
        nf_s.append(dxs.sparcity.mean())
    ind = np.arange(len(label)) 
    width = 0.15
    plt.title('NGram Sparsity')
    plt.bar(ind-width/2,nf_s, width, label='stacked')
    plt.bar(ind+width/2,nf_ns, width, label='unstacked')
    plt.xticks(ind+width, label)
    plt.xlabel("ngrams")
    plt.ylabel("matrix sparcity")
    plt.legend()
    showOrSave(output)


def sparse(f1, output=None, f2=None):
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
    showOrSave(output)

def naiveBayesSmoothing(f1, output=None):
    d1 = pd.read_csv(f1)
    spliton="MultiNB"
    d1["C"] = [float(s.split(spliton)[-1]) if len(s.split(spliton)) > 1 else 0 for s in d1["Model Name"].values]
    model, x, y, ymin, ymax= ParameterImpact(d1)
    plt.plot(x,y, marker='o', label='mean f1')
    plt.plot(x,ymax, marker='o', label='max f1')
    plt.plot(x,ymin, marker='o', label='min f1')
    plt.xlabel('Laplace Smoothing Parameter')
    plt.ylabel('F1 score')
    plt.title('Naive Bayes Smothing Parameter')
    plt.legend()
    showOrSave(output)

def performanceMatrixSize(f1, output=None, f2=None, limit=0.01):
    limit = 0.01 if limit is None else limit
    d1 = pd.read_csv(f1)
    if f2 is not None:
        d2 = pd.read_csv(f2)
        d1 = d1.append(d2)
    #d1 = d1[d1["Stack"] == False]
    show=["MultiNB0.0", "Logistic", "Logistic Lasso5.0"]
    models = d1["Model Name"].unique()
    for model in models :
        if not model in show: continue
        dm = d1[d1["Model Name"] == model]
        f1, sz = dm.F1.values, dm.sz.values/1e9
        combo = sorted([(sz[i], f1[i]) for i in range(len(f1))], key = lambda x : x[0])
        x = [c[0] for c in combo if c[0] > limit]
        y = [c[1] for c in combo if c[0] > limit]
        plt.plot(x,y, marker='o', label=model)
    plt.legend()
    plt.title("Matrix Size vs F1")
    plt.xlabel("Matrix Size Billions")
    plt.ylabel("F1 Score")
    showOrSave(output)

def performanceSparsity(f1, output=None, f2=None, limit=1.0):
    limit = 1.00 if limit is None else limit
    d1 = pd.read_csv(f1)
    if f2 is not None:
        d2 = pd.read_csv(f2)
        d1 = d1.append(d2)
    #d1 = d1[d1["Stack"] == False]
    show=["MultiNB0.0", "Logistic", "Logistic Lasso5.0"]
    models = d1["Model Name"].unique()
    for model in models :
        if not model in show: continue
        dm = d1[d1["Model Name"] == model]
        f1, sz = dm.F1.values, dm.sparcity.values
        combo = sorted([(sz[i], f1[i]) for i in range(len(f1))], key = lambda x : x[0])
        x = [c[0] for c in combo if c[0] < limit]
        y = [c[1] for c in combo if c[0] < limit]
        plt.plot(x,y, marker='o', label=model)
    plt.legend()
    plt.title("Matrix Sparsity vs F1")
    plt.xlabel("Matrix Sparsity")
    plt.ylabel("F1 Score")
    showOrSave(output)


def modelRanking(f1, output=None, f2=None):
    d1 = pd.read_csv(f1)
    if f2 is not None:
        d2 = pd.read_csv(f2)
        d1 = d1.append(d2)
    ngram = d1["NGram"].unique()
    combo = []
    exclude = ["6:6", "8:8", "10:10", "12:12", "20:20", "4:6", "10:15", "15:15"]
    for n in ngram:
        if n in exclude: continue
        dn = d1[d1["NGram"] == n]
        ds = dn[dn["Stack"] == True]
        dns = dn[dn["Stack"] == False]
        combo.append((n, dns.F1.max(), ds.F1.max()))
    combo = sorted(combo, key = lambda x : max(x[1],x[2]))
    label = [c[0] for c in combo]
    vns    = [c[1] for c in combo]
    vs   = [c[2] for c in combo]
    width = 0.15
    ind = np.arange(len(label)) 
    plt.bar(ind-width/2, vs, width, label="F1-stacked")
    plt.bar(ind+width/2, vns, width, label="F1-not-stacked")
    plt.xticks(ind-width, label)
    plt.xlabel("NGram")
    plt.ylabel("F1 Score")
    plt.title("NGram Model Performance")
    plt.legend()
    showOrSave(output)

def createTable(f1, output=None, f2=None):
    d1 = pd.read_csv(f1)
    if f2 is not None:
        d2 = pd.read_csv(f2)
        d1 = d1.append(d2)
    ng = sorted(d1["NGram"].unique(), key = lambda x : int(x.split(':')[0])*10 + int(x.split(':')[1]))    
    ds = d1[d1["Stack"] == True]
    dns = d1[d1["Stack"] == False]
    for i in range(len(ng)):
        dxs = ds[ds['NGram'] == ng[i]]
        dxns = dns[dns['NGram'] == ng[i]]
        if len(dxs) == 0 and len(dxns) == 0: continue
        if len(dxs) == 0 : dxs = dxns
        elif len(dxns) == 0 : dxns = dxs
        bestS = dxs[dxs["F1"] == dxs.F1.max()].iloc[0]["Model Name"]
        bestNS = dxns[dxns["F1"] == dxns.F1.max()].iloc[0]["Model Name"]
        worstS = dxs[dxs["F1"] == dxs.F1.min()].iloc[0]["Model Name"]
        worstNS = dxns[dxns["F1"] == dxns.F1.min()].iloc[0]["Model Name"]
        print("%10s, %10.8f, %10.8f, %7.0f, %7.0f, %10.0f, %10.0f, %7.0f, %7.0f, %6.4f, %6.4f, %6.4f, %6.4f, %s, %s, %s, %s"  % 
              (ng[i], dxns.sparcity.mean(), dxs.sparcity.mean(), dxns.NF.mean(), dxs.NF.mean(), dxns.sz.mean(), dxs.sz.mean(), dxns.nz.mean(), dxs.nz.mean(), dxns.F1.mean(), dxs.F1.mean(),dxns.F1.max(), dxs.F1.max(), bestNS, bestS, worstNS, worstS))


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

    # Size vs F1
    #python analysis.py -i ../analysis/all_lasso.csv ../analysis/naive.csv -r psize -l 1.1 -o SizeVF1

    # Sparsity vs F1
    #python analysis.py -i ../analysis/all_lasso.csv ../analysis/naive.csv -r psparse -l 0.0025 -o SparsityVF1

    # Model Ranking
    #python analysis.py -i ../analysis/all_lasso.csv ../analysis/naive.csv -r rank -o ModelRanking

    parser = argparse.ArgumentParser(description='ML Spring 2019')
    parser.add_argument('-i','--input', nargs='+', default=None)
    parser.add_argument('-o','--output', default=None)
    parser.add_argument('-l','--limit', default=None, type=float)
    parser.add_argument('-r','--run', help='l1 , size, sparse, smooth, psize, psparse', default='l1')
    args = parser.parse_args()

    f1 = args.input[0]
    f2 = args.input[1] if len(args.input) >1 else None

    if args.run == 'l1':
        PlotL1(f1, args.output)
    elif args.run == 'size':
        matrixSize(f1, args.output, f2)
    elif args.run == 'nsparse':
        matrixSparse(f1, args.output, f2)
    elif args.run == 'sparse':
        sparse(f1, args.output)
    elif args.run == 'smooth':
        naiveBayesSmoothing(f1, args.output)
    elif args.run == 'psize':
        performanceMatrixSize(f1, args.output, f2, args.limit)
    elif args.run == 'psparse':
        performanceSparsity(f1, args.output, f2, args.limit) 
    elif args.run == 'rank':
        modelRanking(f1, args.output, f2)
    elif args.run == 'table':
        createTable(f1, args.output, f2)
