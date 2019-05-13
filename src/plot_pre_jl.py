import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys


"""
Compare F1 Scores of the Fits vs NGram Length
"""
def compareFits(ax,df, fits, output, range, title=None, xaxis=None):
    range = int(range) if range is not None  else len(df)
    for f in fits:
        ll = df[df["Model Name"] == f]
        ll = ll[ll["NGramInt"] <= range]
        x = ll.NGramInt
        f1 = ll.F1
        ax.plot(x, f1, marker='o', label='{} F1'.format(f))
    ax.set_ylabel('F1 Score')
    ax.set_title(title)
    if xaxis is not None : ax.set_xlabel(xaxis)
    ax.legend()
    


def getdf(f):
    df = pd.read_csv(f)
    ng = df.NGram.values
    z = [int(x.split(':')[0]) for x in ng]
    df["NGramInt"] = z
    return df
    


if __name__ == '__main__':

    #to see plot
    #python ./plot_pre.py -i ../output/models_stack.csv ../output/models_nostack.csv -r 40

    # to save plot
    #python ./plot_pre.py -i ../output/models_stack.csv ../output/models_nostack.csv -r 40 -o ./combined

    parser = argparse.ArgumentParser(description='ML Spring 2019')
    parser.add_argument('-i','--input', nargs='+',default=None)
    parser.add_argument('-o','--output', default=None)
    parser.add_argument('-d','--direction', default='h')
    parser.add_argument('-r','--range', default=None)
    parser.add_argument('--f0', nargs='+', help= "'svm', 'logistic_lasso', 'Naive_Bayes', 'logistic'", 
                        default=['logistic_lasso', 'Naive_Bayes','nn_40', 'nn_40_40'])
    parser.add_argument('--f1', nargs='+', help= "'svm', 'logistic_lasso', 'Naive_Bayes', 'logistic'", 
                        default=['logistic_lasso', 'Naive_Bayes','svm'])
    args = parser.parse_args()
    assert len(args.input) == 2
    d0 = getdf(args.input[0])
    d1 = getdf(args.input[1])
    output = args.output
    
    if args.direction not in ['h','v']:
      sys.exit("--direction is either 'v' or 'h'")
    elif args.direction == 'h':
      fig, axs=plt.subplots(1,2,figsize=(20,5)) 
      fits0 = [f.replace("_B"," B") for f in args.f0]
      title0 = r'$E[D_t|Agg(St_{t-i}, Min_{t-j}, Sp_{t-k}),NGram]$'
      compareFits(axs[0],d0, fits0, None, args.range, title0, xaxis='NGram Length')

      fits1 = [f.replace("_B"," B") for f in args.f1]
      title1 = r'$E[D_t|St_{t-i}, Min_{t-j}, Sp_{t-k},NGram]$'
      compareFits(axs[1],d1, fits1, None, args.range, title1, xaxis="NGram Length")
      plt.subplots_adjust(hspace=.4)
    
    else:
      fig, axs=plt.subplots(2,1,figsize=(6,6)) 
      fits0 = [f.replace("_B"," B") for f in args.f0]
      title0 = r'$E[D_t|Agg(St_{t-i}, Min_{t-j}, Sp_{t-k}),NGram]$'
      compareFits(axs[0],d0, fits0, None, args.range, title0, xaxis='NGram Length')
      fits1 = [f.replace("_B"," B") for f in args.f1]
      title1 = r'$E[D_t|St_{t-i}, Min_{t-j}, Sp_{t-k},NGram]$'
      compareFits(axs[1],d1, fits1, None, args.range, title1, xaxis="NGram Length")
      plt.subplots_adjust(hspace=.4)
    

    if output is not None:
        plt.savefig("{}.pdf".format(output), bbox_inches='tight')
    else:
        plt.show()






