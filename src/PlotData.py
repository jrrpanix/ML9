import sys
import os
from matplotlib.dates import DateFormatter
import datetime
import matplotlib.pyplot as plt
import numpy as np


class RateDecision:

    def __init__(self, d0, d1, r0, r1, dec, d, dr):
        self.date0 = d0
        self.date1 = d1
        self.rate0 = r0
        self.rate1 = r1
        self.decision = dec
        self.direction = d
        self.dRate = dr

    def __str__(self):
        return "%s,%s,%.2f,%.2f,%s,%d,%.2f" % \
            (self.date0.strftime("%Y%m%d"),self.date1.strftime("%Y%m%d"),self.rate0,
             self.rate1,self.decision,self.direction,self.dRate)
    


class ParseWiki :
    """
    quick code to paree the WikiPedia Fed Rate History Table
    and create a csv file
    input  ../text/history/WikipediaFF.txt
    output ../test/history/WikipediaFFParsed.csv
    """

    def parse(fname="../text/history/WikipediaFF.txt"):
        #output ../test/history/WikipediaFFParsed.csv
        def fmtDate(m, d, y):
            months=["January", "February", "March", "April", "May", "June", "July",
                    "August", "September", "October", "November", "December"]
            mi= months.index(m)
            return "%s%02d%02d" % (y, mi+1, int(d))

        # hack for unrecognized character '-' from web
        def getff(ff):
            if len(ff) < 5:
                return ff,ff
            return ff[0:4], ff[5:]

        ss = "date,fflb,ffub,disc_rate"
        with open(fname) as fd:
            for i,line in enumerate(fd):
                if i == 0 : 
                    continue
                line=line.strip()
                line=line.replace("\t", " ")
                line=line.replace(",", " ")
                line=line.replace("  "," ")
                line=line.replace("   "," ")
                tokens=line.split(" ")
                tokens=[t for t in tokens if len(t) > 0]
                if len(tokens) < 4 : continue
                m,d,y = tokens[0], tokens[1], tokens[2]
                ff = tokens[3].replace("%","")
                fflb, ffub = getff(ff)
                dr = tokens[4].replace("%","")
                ss = "%s,%.2f,%.2f,%s" % (fmtDate(m,d,y), float(fflb),float(ffub), dr)
                print(ss)


class HistDataReader:

    def readWikiCSV(fname, cutoff=datetime.datetime(year=2001, month=1, day=1, hour=0,minute=0,second=0)):
        # read the parsed Wikipedia data "WikipediaFFParsed.csv"
        # input ../test/history/WikipediaFFParsed.csv
        # output returns date, rates vector
        def DT(ds):
            return datetime.datetime(int(ds[0:4]),int(ds[4:6]),int(ds[6:]),0, 0, 0)
        dates,rates = [],[]
        with open(fname) as fd:
            for i,line in enumerate(fd):
                line=line.strip()
                dt=DT(line.split(",")[0])
                if dt < cutoff : continue
                dates.append(dt)
                rates.append(float(line.split(",")[1]))
        return dates,rates

    def readMacroTrends(fname, cutoff=datetime.datetime(year=2001, month=1, day=1, hour=0,minute=0,second=0)):
        # read the downloaded macrotrends csv file
        # input ../text/history/fed-funds-rate-historical-chart.csv
        # output returns date, rates vector
        def DT(year, month, day, hour=0, minute=0, second=0):
            return datetime.datetime(year, month, day, hour, minute, second, 0)

        def parseDate(d):
            return DT(int(d[0:4]), int(d[5:7]), int(d[8:]))
            
        with open(fname) as fd:
            start=False
            dates,rates = [],[]
            for i,line in enumerate(fd):
                line=line.strip()
                if len(line) == 0 : continue
                if not start and len(line.split(",")) != 2 : continue
                if not start and len(line.split(",")) == 2 : 
                    start=True
                    continue
                dt = parseDate(line.split(",")[0])
                if dt < cutoff : continue
                dates.append(dt)
                rates.append(float(line.split(",")[1]))
        return dates,rates
    
    def getMinutesDates(dirname):
        # extract dates from file names of fedminutes data
        # input ../text/minutes - directory name holding minutes files
        # output returns mdates, adates - the two dates on teh minutes file names
        def DT(ds):
            return datetime.datetime(int(ds[0:4]),int(ds[4:6]),int(ds[6:]),0, 0, 0)

        mdates, adates = [], []
        for f in sorted(os.listdir(dirname)):
            se = os.path.splitext(f)[0]
            d0, d1 = se.split("_")[0], se.split("_")[-1]
            mdates.append(DT(d0))
            adates.append(DT(d1))
        return mdates, adates
        

    def readDecision(fname):
        def DT(ds):
            return datetime.datetime(int(ds[0:4]),int(ds[4:6]),int(ds[6:]),0, 0, 0)
        data = []
        with open(fname) as fd:
            for line in fd:
                line = line.strip()
                if len(line) < 2 : continue
                s = line.split(",")
                data.append(RateDecision(DT(s[0]),DT(s[1]),float(s[2]),float(s[3]),s[4],int(s[5]),float(s[6])))
        return data


class PlotFFHist:
    """
    Plot MacroTrends FF, WikiPedia FF and FF minutes dates
    """

    def plotData(macroCSV, wikiCSV, minutesDIR):
        dates1, rates1 = HistDataReader.readMacroTrends(macroCSV)
        dates2, rates2 = HistDataReader.readWikiCSV(wikiCSV)
        mdates, adates =  HistDataReader.getMinutesDates(minutesDIR)
        #Fmt = DateFormatter("%Y-%m")
        Fmt = DateFormatter("%Y")
        fig, ax = plt.subplots()
        ax.set(title="Federal Funds Rate History")
        ax.plot(dates1, rates1, 'b', label='MacroTrends')
        ax.plot(dates2, rates2, 'r', label='WikiPedia')
        MaxRate=np.max(rates1)
        for i in range(len(mdates)):
            ax.plot([mdates[i],mdates[i]], [0, MaxRate], 'g')

        ax.xaxis.set_major_formatter(Fmt)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.show()

class CreateFedAction:

    def create(minutesDIR, macroCSV,tol=0.03):
        # using the minutes dates and the macrotrends FF data set
        # create a dataset which is the FED action

        # input minutesDIR ../text/minutes
        # input macrotrends ../text/history/fed-funds-rate-historical-chart.csv 
        # output RatesDecision.csv
        mdates, adates = HistDataReader.getMinutesDates(minutesDIR)
        dates, rates = HistDataReader.readMacroTrends(macroCSV)
        for i in range(len(mdates)):
            md, ad = mdates[i], adates[i]
            mix = dates.index(md)
            aix = dates.index(ad)
            r0, r1 = rates[mix], rates[aix]
            dr = r1 - r0
            if dr > tol :
                c,d = "raise",1
            elif dr < -tol:
                c,d = "lower",-1
            else:
                c,d = "unchg",0
            print("%s,%s,%.2f,%.2f,%s,%d,%.2f" % 
                  (md.strftime("%Y%m%d"), ad.strftime("%Y%m%d"), rates[mix], rates[aix],c,d,dr))
        
    def plot(decisionFile):
        def points(data, direction):
            dates = np.array([d.date1 for i,d in enumerate(data) if d.direction == direction])
            ddir = np.array([d.direction for d in data if d.direction == direction])
            return dates, ddir
        data = HistDataReader.readDecision(decisionFile)

        update, updir = points(data, 1)
        unchdate, unchdir = points(data, 0)
        dndate, dndir = points(data, -1)
        Fmt = DateFormatter("%Y")
        fig, ax = plt.subplots()
        ax.set(title="Fed Rate Decisions")
        ax.scatter(update, updir, c='g', marker="^")
        ax.scatter(unchdate, unchdir, c='b', marker="x")
        ax.scatter(dndate, dndir, c='r', marker="v")
        ax.xaxis.set_major_formatter(Fmt)
        plt.show()

def main():
    """
    To Run
     python ./PlotData.py 
     defaults : 
     ../text/history/fed-funds-rate-historical-chart.csv 
     ../text/history/WikipediaFFParsed.csv 
     ../text/minutes
     ../text/history/RatesDecision.csv
    """
    defaults=["../text/history/fed-funds-rate-historical-chart.csv", 
              "../text/history/WikipediaFFParsed.csv", 
              "../text/minutes",
              "../text/history/RatesDecision.csv"
              ]
    if len(sys.argv) == 2 and sys.argv[1] =="--help":
        print("Requires 3 Parameters, example usage")
        print("python ./PlotData.py %s %s %s" % (defaults[0], defaults[1], defaults[2]))
        quit()
    elif len(sys.argv) == 4:
        hist1, hist2, dirname = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        hist1, hist2, dirname = defaults[0], defaults[1], defaults[2]
    assert os.path.exists(hist1) and os.path.exists(hist2) and os.path.exists(dirname)
    #PlotFFHist.plotData(hist1, hist2, dirname)
    CreateFedAction.plot(defaults[3])
    

if __name__ == '__main__':
    main()
