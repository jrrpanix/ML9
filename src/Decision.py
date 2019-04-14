import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime 
import numpy as np
from matplotlib.dates import DateFormatter


def getDecisionDF(decisionFile):
    names = ["minutes_date","publish_date","before","after","decision","flag","change"]
    usecols = [0,1,2,3,4,5,6]
    dtypes={"minutes_date":'str',"publish_date":'str',"before":'float',"after":'float',"decision":'str',"flag":'float',"change":'float'}
    df = pd.read_csv(decisionFile, 
                     usecols=usecols,
                     header=None, 
                     names=names,
                     dtype=dtypes,
                     sep=",")
    df['minutes_date'] = pd.to_datetime(df['minutes_date'],format="%Y%m%d")
    df['publish_date'] = pd.to_datetime(df['publish_date'],format="%Y%m%d")
    return df

def PlotDecisionResponse(df):
    Fmt = DateFormatter("%Y")
    u = df[df['decision']=='raise']
    d = df[df['decision']=='lower']
    z = df[df['decision']=='unchg']
    ux, uy = u["publish_date"].values, u['change'].values
    dx, dy = d["publish_date"].values, d['change'].values
    zx, zy = z["publish_date"].values, z['change'].values
    fig, ax = plt.subplots()
    ax.scatter(ux, uy, c='g', marker="^")
    ax.scatter(dx, dy, c='b', marker="x")
    ax.scatter(zx, zy, c='r', marker="v")
    ax.xaxis.set_major_formatter(Fmt)
    ax.set(title="Rate Decision Response")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Spring 2019 Final Project')
    parser.add_argument('-i','--decisionFile', help='directory holding minutes dates', default="../text/history/RatesDecision.csv")
    parser.add_argument('-r','--fedFundsHistFile', help='file holding historcal fed runds rate',
                        default="../text/history/FedFundsRates.csv")
    parser.add_argument('-o','--output', help='output file', default=None)
    parser.add_argument('-p','--plot', action='store_true', default=False) 
    args = parser.parse_args()
    
    df=getDecisionDF(args.decisionFile)
    PlotDecisionResponse(df)
