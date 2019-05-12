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
            'F1':['mean']})
    mn = g['Model Name']['<lambda>'].values
    gv = g['F1']['mean'].values
    cv = g['C']['mean'].values
    mn =[s.split(" ")[-1] for s in mn]
    combo=sorted([(m,vx,ci) for m,vx,ci in zip(mn, gv, cv)], key=lambda x : x[2], reverse=False)
    reg = np.zeros(len(combo))
    f1 = np.zeros(len(combo))
    for i,c in enumerate(combo):
        reg[i] = c[2]
        f1[i] = c[1]
    return reg, f1

#
# Plot impact of regularization parameter on predicted F1 Score
#    
def PlotL1(f1, output=None):
    d1 = pd.read_csv(f1)
    dns = d1[d1["Stack"] == False]
    xns, yns = L1_Impact(dns)

    ds = d1[d1["Stack"] == True]
    xs, ys = L1_Impact(ds)
    
    plt.title('Logistic Lasso Regularization')
    plt.xlabel('L1 Regularization, larger more regularization')
    plt.ylabel('F1 Score')

    plt.plot(xns[1:],yns[1:], marker='o', label='more sparsity')
    plt.plot(xs[1:],ys[1:], marker='o', label='less sparsity')
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

if __name__ == '__main__':
    # to get bar graph of matrix sizes
    # python ./analysis.py -i ../analysis/all_lasso.csv -r size -o [to save to file]
    #

    # to get bar graph of matrix sizes
    # python ./analysis.py -i ../analysis/all_lasso.csv -r size
    #

    parser = argparse.ArgumentParser(description='ML Spring 2019')
    parser.add_argument('-i','--input', default=None)
    parser.add_argument('-o','--output', default=None)
    parser.add_argument('-r','--run', help='l1 , size', default='l1')
    args = parser.parse_args()

    if args.run == 'l1':
        # ../analysis/all_lasso.csv
        PlotL1(args.input, args.output)
    elif args.run == 'size':
        matrixSize(args.input, args.output)

