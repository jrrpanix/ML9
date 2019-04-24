import os
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

#
# Code simulator

#
# This run random experiments to detect sentimate in randomly generated text
#
# Overview 
# 1) load 3 word dictionaries common, positive, negative
# 2) randomly generate data using (common, positive) (common, negative), shuffle
# 3) split into test training
# 4) create features (X's) for trainng/test using Vectorize
# 5) fit the model
# 6) check its accuracy

# To run download the following 3 files of common, positive and negative words: 
# (1) A list of 20k commonly used English Words
#    curl https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt -o 20k.txt
# (2) A list of negative words 
#    curl https://gist.githubusercontent.com/mkulakowski2/4289441/raw/dad8b64b307cd6df8068a379079becbb3f91101a/negative-words.txt -o neg_words.tx
# (3) A list of positive words
#    curl https://gist.githubusercontent.com/mkulakowski2/4289437/raw/1bb4d7f9ee82150f339f09b5b1a0e6823d633958/positive-words.txt -o pos_words.txt
# To run download the following 3 files
#

#
# read in words from common, postive, negative
# return 3 lists common, postive, negative
#

class Words:

    def __init__(self, common, neg, pos):
        self.common = common
        self.neg = neg
        self.pos = pos

    def load(commonFile, negFile, posFile): 
        neg = [w for w in open(negFile).read().split('\n')
               if len(w) > 0 and w[0] != ';']
        
        pos = [w for w in open(posFile).read().split('\n')
               if len(w) > 0 and w[0] != ';']
        common = [w for w in open(commonFile).read().split('\n')
                  if len(w) > 0]
    
        return Words(common, neg, pos)

class Params:

    def __init__(self, trials, nSamples, textLen, pctSen, pctTrain, max_iter, solver):
        self.trials = trials
        self.nSamples = nSamples
        self.textLen = textLen
        self.pctSen = pctSen 
        self.pctTrain = pctTrain
        self.max_iter = max_iter
        self.solver = solver


class DataGen:
    # generateData
    # inputs
    # N       - total sample size generated
    # Ncommon - number of common_words in text
    # Nsen    - number of sentiment words in text
    # common_words - list of common words
    # pos_words - list of positive words
    # neg_words - list of negative words
    # output 
    # returns randomly shuffeled nparray ([text0, score0], [text1, score1], ... [textN-1, scoreN-1])
    # where text_i is either postive or negative
    def gen(words, params):
        def genText(Ncommon, common_words, Nsen, sen_words):
            words=np.concatenate((np.random.choice(sen_words, Nsen),
                                  np.random.choice(common_words, Ncommon)))
            np.random.shuffle(words)
            return " ".join(words[i] for i in range(len(words)))
        pos_score, neg_score = 1,0
        Nsen = int(params.textLen*params.pctSen)
        Ncommon = params.textLen - Nsen
        pos_data = [[genText(Ncommon, words.common, Nsen, words.pos),pos_score] 
                    for i in range(params.nSamples)]
        neg_data = [[genText(Ncommon, words.common, Nsen, words.neg),neg_score] 
                    for i in range(params.nSamples)]
        all_data = np.concatenate((pos_data, neg_data))
        np.random.shuffle(all_data)
        Ntrain = int(len(all_data)*params.pctTrain)
        train_data = pd.DataFrame(all_data[0:Ntrain],columns=['text', 'sentiment'])
        test_data = pd.DataFrame(all_data[Ntrain:], columns=['text', 'sentiment'])
        return train_data, test_data


class Simulator:

    def trial(words, params, models, resdict, i):
        train_data, test_data = DataGen.gen(words, params)
        vectorizer = CountVectorizer(stop_words="english",preprocessor=None)
        training_features = vectorizer.fit_transform(train_data["text"])                                 
        test_features = vectorizer.transform(test_data["text"])
        result={}
        for (name, model) in models:
            model.fit(training_features, train_data["sentiment"])
            y_pred = model.predict(test_features)
            acc = accuracy_score(test_data["sentiment"], y_pred)
            resdict[name][i]=acc
        return pd.DataFrame(result)

    def run(words, params):
        max_iter, solver = params.max_iter, params.solver
        models=[("svm",LinearSVC(max_iter=max_iter)),
                ("logistic",LogisticRegression(solver=solver,max_iter=max_iter)),
                ("logistic_lasso",LogisticRegression(penalty='l1',solver=solver,max_iter=max_iter)),
                ("Naive Bayes",MultinomialNB())]
        resdict = {name:np.zeros(params.trials) for (name,model) in models}
        for i in range(params.trials):
            Simulator.trial(words, params, models, resdict, i)
        return pd.DataFrame(resdict)

def makePlot():
    plt.plot(L, Asvm, label='svm_{}'.format(label))
    plt.plot(L, Alog, label='logistic_{}'.format(label))
    plt.plot(L, Aloglasso, label='logistic_lasso_{}'.format(label))
    plt.title("Accuracy vs N Words in Text")
    plt.xlabel("Words in Text")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Project')
    parser.add_argument('--common', default="../simdata/20k.txt")
    parser.add_argument('--positive', default="../simdata/pos_words.txt")
    parser.add_argument('--negative', default="../simdata/neg_words.txt")
    parser.add_argument('-n','--nSamples', default=50, type=int)
    parser.add_argument('-t','--trials', default=3, type=int)
    parser.add_argument('-l','--textLen', default=60, type=int)
    parser.add_argument('-s','--pctSen', help='pct of sentiment words in the text', default=0.10, type=float)
    parser.add_argument('-p','--pctTrain', help='pct of sample for training', default=0.75, type=float)
    parser.add_argument('--stp', help='stepsize', default=10, type=int)
    parser.add_argument('--max_iter', help='max iterations', default=100, type=int)
    parser.add_argument('--solver', help="solver for sklearn algo", default='liblinear')
    args = parser.parse_args()


    words = Words.load(args.common, args.negative, args.positive)
    params = Params(args.trials, args.nSamples, args.textLen, args.pctSen, args.pctTrain, args.max_iter, args.solver)
    res = Simulator.run(words, params)
    print(res)




