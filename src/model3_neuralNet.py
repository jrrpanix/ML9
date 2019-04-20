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
import torch.utils.data as data 
from torch.autograd import Variable


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

    net = FCNet1(100)
    

