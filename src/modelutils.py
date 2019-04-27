import pandas as pd
import os
import datetime
import numpy as np
import bisect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


"""
put repeated stuff here that is used in building the models
"""

class modelutils:

    
    def decisionDF(decisionFile):
        names = ["MeetingDate",
                 "MinutesRelease",
                 "PreviousTarget",
                 "PostTarget",
                 "Direction",
                 "ActionFlag",
                 "Amount"]
        usecols = [0,1,2,3,4,5,6]
        dtypes={"MeetingDate":'str',
                "MinutesRelease":'str',
                "PreviousTarget":'float',
                "PostTarget":'float',
                "Direction":'str',
                "ActionFlag":'float',
                "Amount":'float'}
        df = pd.read_csv(decisionFile, 
                         usecols=usecols,
                         header=None, 
                         names=names,
                         dtype=dtypes,
                         sep=",")
        # done outside of df construction as dateimte conversion can be extremely 
        # slow without a format hint as data size grows
        df['MeetingDate'] = pd.to_datetime(df['MeetingDate'],format="%Y%m%d")
        df['MinutesRelease'] = pd.to_datetime(df['MinutesRelease'],format="%Y%m%d")
        return df

    def updateData(data, minutesReleaseDate, docDate, docText, action, amount, direction, docType):
        data["MinutesRelease"].append(minutesReleaseDate)
        data["DocDate"].append(docDate)
        data["Year"].append(docDate.year)
        data["DocText"].append(docText)
        data["ActionFlag"].append(action)
        data["Amount"].append(amount)
        data["Direction"].append(direction)
        data["DocumentType"].append(docType)

    def getDataCols():
        return {"MinutesRelease":[],"DocDate":[], "Year":[], "DocText":[], "ActionFlag":[], "Amount":[], "Direction":[], "DocumentType":[]}

    """
    return FedMinutes as pandas DataFrame
    """
    def getMinutes(minutesDir, df, clean_algo, abortOnFail=False):
        data = modelutils.getDataCols()
        for files in sorted(os.listdir(minutesDir)):
            f, ext = os.path.splitext(files)
            try:
                # commented out for now as its not used 
                #minutes_date = datetime.datetime.strptime(f.split("_")[0],"%Y%m%d")
                release_date = datetime.datetime.strptime(f.split("_")[-1],"%Y%m%d")
                text = clean_algo(open(os.path.join(minutesDir, files)).read().strip())
                action = df[df["MinutesRelease"] == release_date].iloc[0]["ActionFlag"]
                amount = df[df["MinutesRelease"] == release_date].iloc[0]["Amount"]
                direction = df[df["MinutesRelease"] == release_date].iloc[0]["Direction"] 
                modelutils.updateData(data, release_date, release_date, text, action, amount, direction, "minutes")
            except Exception as e:
                print("exception reading minutes, file %s" % files)
                print(e)
                if abortOnFail:
                    quit()
        return pd.DataFrame(data)


    def getStatements(statementsDir, df, clean_algo, abortOnFail=False):
        data = modelutils.getDataCols()
        for files in sorted(os.listdir(statementsDir)):
            f, ext = os.path.splitext(files)
            try:
                statement_date = datetime.datetime.strptime(f.split(".")[0],'%Y%m%d') 
                ix = modelutils.get_ix(df["MinutesRelease"].values, statement_date)
                release_date = df.loc[ix]["MinutesRelease"]
                action = df.loc[ix]["ActionFlag"]
                amount = df.loc[ix]["Amount"]
                direction = df[df["MinutesRelease"] == release_date].iloc[0]["Direction"]
                text = clean_algo(open(os.path.join(statementsDir,files),encoding='utf-8',errors='ignore').read().strip())
                modelutils.updateData(data, release_date, statement_date, text, action, amount, direction, "statements")
            except Exception as e:
                print("exception reading statements, file %s" % files)
                print(e)
                if abortOnFail:
                    quit()
        return pd.DataFrame(data)


    def getSpeeches(speechesDir, df, clean_algo, abortOnFail=False):
        data = modelutils.getDataCols()
        for files in sorted(os.listdir(speechesDir)):
            f, ext = os.path.splitext(files)
            try:
                speech_date = datetime.datetime.strptime(f.split("_")[0],'%Y%m%d') 
                ix = modelutils.get_ix(df["MinutesRelease"].values, speech_date)
                release_date = df.loc[ix]["MinutesRelease"]
                action = df.loc[ix]["ActionFlag"]
                amount = df.loc[ix]["Amount"]
                direction = df[df["MinutesRelease"] == release_date].iloc[0]["Direction"]
                text = clean_algo(open(os.path.join(speechesDir,files),encoding='utf-8',errors='ignore').read().strip())
                modelutils.updateData(data, release_date, speech_date, text, action, amount, direction, "speeches")
            except Exception as e:
                print("exception reading statements, file %s" % files)
                print(e)
                if abortOnFail:
                    quit()
        return pd.DataFrame(data)

    def get_ix(dateV, date):
        dt64 = modelutils.to_np64(date)
        return modelutils.get_ge(dateV, dt64)

    def get_ge(dateV, date):
        return bisect.bisect_right(dateV, date)

    def get_lt(dateV, date):
        return bisect.bisect_left(dateV, date) - 1

    def to_dt(date): # convert a np.datetime64 to a datetime.datetime
        if type(date) == datetime.datetime : return date
        ts = (date - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        return datetime.datetime.utcfromtimestamp(ts)

    def to_np64(date):
        return np.datetime64(date)

    def to_days(x):
        return x.astype('timedelta64[D]')/np.timedelta64(1, 'D')

    def splitTrainTest(data_sets, train_pct):
        combined = pd.concat(data_sets) if type(data_sets) is list else data_sets
        return train_test_split(combined, test_size=1.0-train_pct)

    def getFeatures(train_data, test_data, ngram):
        vectorizer = CountVectorizer(stop_words="english",preprocessor=None, ngram_range=ngram)
        training_features = vectorizer.fit_transform(train_data["DocText"])                                 
        test_features = vectorizer.transform(test_data["DocText"])
        return training_features, test_features

    def getBounds(datadf):
        return (len(datadf), modelutils.to_dt(datadf["DocDate"].min()), modelutils.to_dt(datadf["DocDate"].max()))
