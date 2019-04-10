from xml.dom.minidom import parse
from array import array
from urllib.request import urlopen
from xml.dom import minidom
from io import StringIO
from lxml import etree
from dateutil import parser
from matplotlib.backends.backend_pdf import PdfPages
from datetime import date

import datetime as dt
import xml.etree.ElementTree as ET
import json, requests
import xmltodict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv


def plot_graph(x,y,name):
    fig = plt.figure()
    fig.suptitle(name, fontsize=14, fontweight='bold')
    plt.plot(datelist, data_array)
    plt.gcf().autofmt_xdate()
    return fig

api_key = input("Input API Key: ")
api_key = '&api_key='+api_key

csv_flag = int(input("Create CSV files (1=yes, 2=no): "))
number_series = int(input("How many series to analyze: ") )

series_name_url = {}
series_name_string = {}
json_map = {}
time_string = date.today().strftime("%Y-%m-%d")
pp = PdfPages('FRED_Charts_'+time_string+'.pdf')

for x in range (0, number_series):
    
    series = input("What series: ")
    start_date = input("Start Date (mm/dd/yyyy): ")
    end_date = input("End Date (mm/dd/yyyy - '12/31/9999' for latest): ")
    start_date = dt.datetime.strptime(start_date,"%m/%d/%Y").strftime("%Y-%m-%d")
    end_date = dt.datetime.strptime(end_date,"%m/%d/%Y").strftime("%Y-%m-%d")
    FRED = 'https://api.stlouisfed.org/fred/series/observations?'
    xmlurl = FRED+ 'series_id='+series+'&observation_start=' + start_date + '&observation_end=' + end_date + api_key +'&file_type=json'
    series_name_url[series] = xmlurl
    series_name_string[x] = series
    json_map[series] = requests.get(xmlurl)

for x in range (0, number_series):
    temp = series_name_string[x]
    
    json_string = json_map[series_name_string[x]]

    data = json.loads(json_string.text)

    number = data["count"]
    data_array = pd.Series(number)
    date_list = [number]

    for y in range (0,number):
        if data["observations"][y]["value"] =='.':
            date_list.append(data["observations"][y]["date"])
            data_array[y] = np.nan 

        else:
            date_list.append(data["observations"][y]["date"])
            data_array[y] = float(data["observations"][y]["value"])


    dates = dt.datetime.date(dt.datetime.strptime(date_list[1], "%Y-%m-%d"))

    datelist = pd.date_range(pd.datetime.today(), periods = number).tolist()

    for z in range (0,number):
        tempdate = dt.datetime.strptime(date_list[z+1], "%Y-%m-%d")
        datelist[z] =dt.datetime.date(tempdate)
        
    if csv_flag == 1:
        myfile = open(series_name_string[x]+'.csv', 'wt', newline='')
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow( ('date', series_name_string[x]))

        for a in range (0,number):
            wr.writerow( (datelist[a], data_array[a]) )
            
        myfile.close()
        
    plot = plot_graph(datelist, data_array, temp)
    pp.savefig(plot)
    plot.show()

pp.close()


