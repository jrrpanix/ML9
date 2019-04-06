import sys
import os

"""
quick code to paree the WikiPedia Fed Rate History Table
and create a csv file

python ./parseHist.py ../text/history/hist_wikipedia.txt 
"""

def fmtDate(m, d, y):
    months=["January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"]
    mi= months.index(m)
    return "%s%02d%02d" % (y, mi+1, int(d))

def getff(ff):
    if len(ff) < 5:
        return ff,ff
    # hack for unrecognized character '-' from web
    return ff[0:4], ff[5:]


def filterData(fname):
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


def main():
    fname = sys.argv[1]
    assert os.path.exists(fname)
    filterData(fname)

if __name__ == '__main__':
    main()
