import pandas as pd

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

