from xml.dom.minidom import parse
from array import array
from urllib.request import urlopen
from xml.dom import minidom
from io import StringIO
from lxml import etree
from dateutil import parser

import datetime as dt
import xml.etree.ElementTree as ET
import json, requests
import xmltodict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv
import os
import sys

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

path = "../../text/history"

if not os.path.exists(path):
    os.mkdir(path)

    

api_key = input("Input API Key: ")
api_key = '&api_key='+api_key

number_series = int(input("How many series to analyze: ") )

series_name_url = {}
series_name_string = {}
json_map = {}

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
        tempdate = tempdate.strftime("%Y%m%d")
        datelist[z] =tempdate #dt.datetime.date(tempdate)

    myfile = open(path+'/'+series_name_string[x]+'.csv', 'wt', newline='')
    wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
    wr.writerow( ('date', series_name_string[x]))

    for x in range (0,number):
        wr.writerow( (datelist[x], data_array[x]) )

    myfile.close()

    fig = plt.figure()
    fig.suptitle(temp, fontsize=14, fontweight='bold')

    plt.plot(datelist, data_array)
    plt.gcf().autofmt_xdate()
