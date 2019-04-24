import os
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

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
def loadWords(commonFile="./20k.txt", positiveFile="./pos_words.txt", negativeFile="./neg_words.txt"):
    # curl https://gist.githubusercontent.com/mkulakowski2/4289441/raw/dad8b64b307cd6df8068a379079becbb3f91101a/negative-words.txt -o neg_words.tx
    # curl https://gist.githubusercontent.com/mkulakowski2/4289437/raw/1bb4d7f9ee82150f339f09b5b1a0e6823d633958/positive-words.txt -o pos_words.txt
    # curl https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt -o 20k.txt

    neg = [w for w in open(negativeFile).read().split('\n')
           if len(w) > 0 and w[0] != ';']

    pos = [w for w in open(positiveFile).read().split('\n')
           if len(w) > 0 and w[0] != ';']

    common = [w for w in open(commonFile).read().split('\n')
              if len(w) > 0]
    
    return common, pos, neg

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
def generateData(common_words, pos_words, neg_words, N, textLen, pctSen, pos_score=1, neg_score=0):
    def genText(Ncommon, common_words, Nsen, sen_words):
        words=np.concatenate((np.random.choice(sen_words, Nsen), np.random.choice(common_words, Ncommon)))
        np.random.shuffle(words)
        return " ".join(words[i] for i in range(len(words)))
    Nsen = int(textLen*pctSen)
    Ncommon = textLen - Nsen
    pos_data = [[genText(Ncommon, common_words, Nsen, pos_words),pos_score] for i in range(N)]
    neg_data = [[genText(Ncommon, common_words, Nsen, neg_words),neg_score] for i in range(N)]
    all_data = np.concatenate((pos_data, neg_data))
    np.random.shuffle(all_data)
    return all_data

# splitData
# input - the output from generateData
# output -  pandas DataFrame columns=['text', 'sentiment']
def splitData(all_data, pctTrain):
    Ntrain = int(len(all_data)*pctTrain)
    train_data = pd.DataFrame(all_data[0:Ntrain],columns=['text', 'sentiment'])
    test_data = pd.DataFrame(all_data[Ntrain:], columns=['text', 'sentiment'])
    return train_data, test_data

def experimentData(common, pos, neg, N, textLen, pctSen, pctTrain):
    all_data = generateData(common, pos, neg, N, textLen, pctSen)
    train_data, test_data = splitData(all_data, pctTrain)
    return train_data, test_data

def fitData(train_data, test_data, modelType="svc"):
    
    # create features
    vectorizer = CountVectorizer(stop_words="english",preprocessor=None)
    training_features = vectorizer.fit_transform(train_data["text"])                                 
    test_features = vectorizer.transform(test_data["text"])

    # run model
    if modelType == "logistic":
        model = LogisticRegression()
    elif modelType == "logistic_lasso":
        model = LogisticRegression(penalty='l1')
    else:
        model = LinearSVC()
    model.fit(training_features, train_data["sentiment"])
    y_pred = model.predict(test_features)
    acc = accuracy_score(test_data["sentiment"], y_pred)
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Project')
    parser.add_argument('--common', default="./20k.txt")
    parser.add_argument('--positive', default="./pos_words.txt")
    parser.add_argument('--negative', default="./neg_words.txt")
    parser.add_argument('-n','--N', default=50, type=int)
    parser.add_argument('-t','--trials', default=3, type=int)
    parser.add_argument('-l','--textLen', default=60, type=int)
    parser.add_argument('-s','--pctSen', help='pct of sentiment words in the text', default=0.10, type=float)
    parser.add_argument('-p','--pctTrain', help='pct of sample for training', default=0.75, type=float)
    args = parser.parse_args()


    common, pos, neg = loadWords(args.common, args.positive, args.negative)
    T, N, textLen, pctSen, pctTrain = args.trials, args.N, args.textLen, args.pctSen, args.pctTrain
    
    print("%10s, %10s, %10s, %10s, %10s, %10s, %10s" %('T','N',' TextLen', 'PctSen', 'PctTrain', 'mean(acc)', 'std(acc)'))
    stp=40
    S = int(textLen/stp)
    L, Asvm, Alog, Aloglasso = np.zeros(S), np.zeros(S), np.zeros(S), np.zeros(S)
    # vary textLen
    for j, l in enumerate(range(stp,textLen+1,stp)):
        accsvm, acclog, accloglasso = np.zeros(T), np.zeros(T), np.zeros(T)
        for i in range(T):
            train_data, test_data = experimentData(common, pos, neg, N, l, pctSen, pctTrain)
            accsvm[i] = fitData(train_data, test_data, modelType='svc')
            acclog[i] = fitData(train_data, test_data, modelType='logistic')
            accloglasso[i] = fitData(train_data, test_data, modelType='logistic_lasso')
        Asvm[j] = np.mean(accsvm)
        Alog[j] = np.mean(acclog)
        Aloglasso[j] = np.mean(accloglasso)
        L[j] = l
        print("%10d, %10d, %10d, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f" % 
              (T,N, l, pctSen, pctTrain, np.mean(accsvm), np.std(accsvm), np.mean(acclog), np.std(acclog), np.mean(accloglasso), np.std(accloglasso)))
            

    plt.plot(L, Asvm, label='svm')
    plt.plot(L, Alog, label='logistic')
    plt.plot(L, Aloglasso, label='logistic_lasso')
    plt.title("Accuracy vs N Words in Text")
    plt.xlabel("Words in Text")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

