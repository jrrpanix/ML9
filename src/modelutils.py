import pandas as pd
import os
import datetime

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
        minutes, publish , data = [], [], []
        for files in sorted(os.listdir(minutesDir)):
            f, ext = os.path.splitext(files)
            minutes.append(datetime.datetime.strptime(f.split("_")[0],"%Y%m%d"))
            publish.append(datetime.datetime.strptime(f.split("_")[-1],"%Y%m%d"))
            try:
                text = clean_algo(open(os.path.join(minutesDir, files)).read().strip())
                dec = df[df["publish_date"] == publish[-1]].iloc[0]["flag"]
                data.append([text, dec])
            except Exception as e:
                print("exception reading minutes, file %s" % files)
                print(e)
                if abortOnFail:
                    quit()
            return data, publish


    def getStatements(statementsDir, df, clean_algo, abortOnFail=False):
        statements, data = [], []
        for files in sorted(os.listdir(statementsDir)):
            f, ext = os.path.splitext(files)
            statements.append(datetime.datetime.strptime(f.split(".")[0],'%Y%m%d'))
            try:
                text = clean_algo(open(os.path.join(statementsDir,files),encoding='utf-8',errors='ignore').read().strip())
                dec = df[df["publish_date"] == statements[-1]].iloc[0]["flag"]
                data.append([text,dec])
            except Exception as e:
                print("exception reading statements, file %s" % files)
                print(e)
                if abortOnFail:
                    quit()
        return data

