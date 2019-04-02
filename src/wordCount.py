import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk import FreqDist
import nltk
import sys

"""
 to install nltk stuff
 python - great a pyton repl
 >>> import nltk
 >>> nltk.download()
"""

def sampleCount():
    texts = ['hi there', 'hello there', 'hello here you are']
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    freq = np.ravel(X.sum(axis=0))
    freq.plot()
    print(freq)

def sampleFreqDist(texts):
    #texts = 'hi there hello there'
    words = nltk.tokenize.word_tokenize(texts)
    fdist = FreqDist(words)
    fdist.plot(30)

def getData(fname):
    with open(fname) as f:
        text = f.read()
    return text.replace("\n", " " )

def main():
    fname = sys.argv[1]
    texts = getData(fname)
    sampleFreqDist(texts)

if __name__ == '__main__':
    main()

