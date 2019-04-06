import sys
import os
from matplotlib.dates import DateFormatter
import datetime
import matplotlib.pyplot as plt
import numpy as np

"""
quick code to paree the WikiPedia Fed Rate History Table
and create a csv file

python ./parseHist.py ../text/history/hist_wikipedia.txt 
"""


class ParseWiki :

    def parse(fname):
        # ../text/history/hist_wikipedia.txt 
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

class PlotFFHist:

    def readData(fname, cutoff=datetime.datetime(year=2001, month=1, day=1, hour=0,minute=0,second=0)):
        # ../text/history/fed-funds-rate-historical-chart.csv
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

    def plotData(fname):
        dates,rates = PlotFFHist.readData(fname)
        Fmt = DateFormatter("%Y-%m")
        fig, ax = plt.subplots()
        ax.set(title="Federal Funds Rate History")
        ax.plot(dates, rates)
        ax.xaxis.set_major_formatter(Fmt)
        plt.show()

def main():
    fname = sys.argv[1]
    assert os.path.exists(fname)
    #ParseWiki.parse(fname)
    # python ./parseHist.py ../text/history/fed-funds-rate-historical-chart.csv
    PlotFFHist.plotData(fname)
    

if __name__ == '__main__':
    main()
