import sys
import os
from matplotlib.dates import DateFormatter
import datetime
import matplotlib.pyplot as plt
import numpy as np



class ParseWiki :
    """
    quick code to paree the WikiPedia Fed Rate History Table
    and create a csv file

    python ./parseHist.py ../text/history/hist_wikipedia.txt 
    """

    def parse(fname):
        # ../text/history/hist_wikipedia.txt > FedHistory.csv
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
    """
    Plot the 2 different sources of FF on same graph
    """

    def parseHist2(fname, cutoff=datetime.datetime(year=2001, month=1, day=1, hour=0,minute=0,second=0)):
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

    def parseHist1(fname, cutoff=datetime.datetime(year=2001, month=1, day=1, hour=0,minute=0,second=0)):
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

    def getMinuteDates(dirname):
        def DT(ds):
            return datetime.datetime(int(ds[0:4]),int(ds[4:6]),int(ds[6:]),0, 0, 0)

        mdates, adates = [], []
        for f in sorted(os.listdir(dirname)):
            se = os.path.splitext(f)[0]
            d0, d1 = se.split("_")[0], se.split("_")[-1]
            mdates.append(DT(d0))
            adates.append(DT(d1))
        return mdates, adates

    def plotData(hist1, hist2, dirname):
        dates1, rates1 = PlotFFHist.parseHist1(hist1)
        dates2, rates2 = PlotFFHist.parseHist2(hist2)
        mdates, adates =  PlotFFHist.getMinuteDates(dirname)
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

def main():
    """
    To Run
     python ./parseHist.py ../text/history/fed-funds-rate-historical-chart.csv ../text/history/FedHistory.csv 
    """
    if len(sys.argv) < 4:
        print("Requires 3 Parameters, example usage")
        print("python ./parseHist.py ../text/history/fed-funds-rate-historical-chart.csv ../text/history/FedHistory.csv ../text/minutes")
        quit()

    hist1, hist2, dirname = sys.argv[1], sys.argv[2], sys.argv[3]
    assert os.path.exists(hist1) and os.path.exists(hist2)
    PlotFFHist.plotData(hist1, hist2, dirname)
    

if __name__ == '__main__':
    main()
