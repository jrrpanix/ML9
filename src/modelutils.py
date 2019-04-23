import pandas as pd
import os
import datetime
import numpy as np
import bisect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


"""
put repeated stuff here that is used in building the models
"""

class modelutils:

    
    def decisionDF(decisionFile):
        names = ["minutes_date",
                 "publish_date",
                 "before",
                 "after",
                 "decision",
                 "flag",
                 "change"]
        usecols = [0,1,2,3,4,5,6]
        dtypes={"minutes_date":'str',
                "publish_date":'str',
                "before":'float',
                "after":'float',
                "decision":'str',
                "flag":'float',
                "change":'float'}
        df = pd.read_csv(decisionFile, 
                         usecols=usecols,
                         header=None, 
                         names=names,
                         dtype=dtypes,
                         sep=",")
        # done outside of df construction as dateimte conversion can be extremely 
        # slow without a format hint as data size grows
        df['minutes_date'] = pd.to_datetime(df['minutes_date'],format="%Y%m%d")
        df['publish_date'] = pd.to_datetime(df['publish_date'],format="%Y%m%d")
        return df


    def getMinutes(minutesDir, df, clean_algo, abortOnFail=False):
        publish , data = [], []
        for files in sorted(os.listdir(minutesDir)):
            f, ext = os.path.splitext(files)
            try:
                # commented out for now as its not used 
                #minutes_date = datetime.datetime.strptime(f.split("_")[0],"%Y%m%d")
                publish_date = datetime.datetime.strptime(f.split("_")[-1],"%Y%m%d")
                text = clean_algo(open(os.path.join(minutesDir, files)).read().strip())
                dec = df[df["publish_date"] == publish_date].iloc[0]["flag"]
                data.append([text, dec])
                publish.append(publish_date)
            except Exception as e:
                print("exception reading minutes, file %s" % files)
                print(e)
                if abortOnFail:
                    quit()
        return data, publish


    def getStatements(statementsDir, df, clean_algo, abortOnFail=False, debug=False):
        cleanedStatementV, dateV = [], df["publish_date"].values
        for files in sorted(os.listdir(statementsDir)):
            f, ext = os.path.splitext(files)
            try:
                statement_date = datetime.datetime.strptime(f.split(".")[0],'%Y%m%d') 
                dt64 = modelutils.to_np64(statement_date)
                ix = modelutils.get_ge(dateV, dt64)
                decision = df.loc[ix]["flag"]
                if debug :
                    print("getStatements debug", df.loc[ix]["publish_date"], ix, statement_date, decision)
                text = clean_algo(open(os.path.join(statementsDir,files),encoding='utf-8',errors='ignore').read().strip())
                cleanedStatementV.append([text,decision])
            except Exception as e:
                print("exception reading statements, file %s" % files)
                print(e)
                if abortOnFail:
                    quit()
        return cleanedStatementV

    def get_ge(datesV, date):
        return bisect.bisect_right(datesV, date)

    def get_lt(datesV, date):
        return bisect.bisect_left(datesV, date) - 1

    def to_dt(date): # convert a np.datetime64 to a datetime.datetime
        if type(date) == datetime.datetime : return date
        ts = (date - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        return datetime.datetime.utcfromtimestamp(ts)

    def to_np64(date):
        return np.datetime64(date)

    def to_days(x):
        return x.astype('timedelta64[D]')/np.timedelta64(1, 'D')


    def splitTrainTest(data_sets, train_pct):
        def isMultipleDataSets(data_sets):
            for d in data_sets:
                # first line is 2 elements and has decision variable
                if len(d) == 2 and d[1] < 3: return False
                return True
        combined = data_sets
        if isMultipleDataSets(data_sets):
            combined = data_sets[0]
            for i in range(1,len(data_sets)):
                combined.extend(data_sets[i])
        Ntrain = int(len(combined)* train_pct)
        np.random.shuffle(combined)
        train_data = pd.DataFrame(combined[0:Ntrain],columns=['text', 'sentiment'])
        test_data = pd.DataFrame(combined[Ntrain:], columns=['text', 'sentiment'])
        return train_data, test_data

    def getFeatures(train_data, test_data, ngram):
        vectorizer = CountVectorizer(stop_words="english",preprocessor=None, ngram_range=ngram)
        training_features = vectorizer.fit_transform(train_data["text"])                                 
        test_features = vectorizer.transform(test_data["text"])
        return training_features, test_features

