from xml.dom.minidom import parse
from array import array
from urllib.request import urlopen
from xml.dom import minidom
from io import StringIO
from lxml import etree
from dateutil import parser

import datetime as dt
import urllib
import xml.etree.ElementTree as ET
import json, requests
import xmltodict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

api_key = input("Input API Key: ")
api_key = '&api_key='+api_key

series = input("What series: ")
start_date = input("Start Date (mm/dd/yyyy): ")
end_date = input("End Date (mm/dd/yyyy - '12/31/9999' for latest): ")
start_date = dt.datetime.strptime(start_date,"%m/%d/%Y").strftime("%Y-%m-%d")
end_date = dt.datetime.strptime(end_date,"%m/%d/%Y").strftime("%Y-%m-%d")

FRED = 'https://api.stlouisfed.org/fred/series/observations?'

xmlurl = FRED+ 'series_id='+series+'&observation_start=' + start_date + '&observation_end=' + end_date + api_key +'&file_type=json'

json_string = requests.get(xmlurl)

data = json.loads(json_string.text)

number = data["count"]
gdp_array = pd.Series(number)
date_list = [number]

for x in range (0,number):
    if data["observations"][x]["value"] =='.':
        date_list.append(data["observations"][x]["date"])
        gdp_array[x] = np.nan 

    else:
        date_list.append(data["observations"][x]["date"])
        gdp_array[x] = float(data["observations"][x]["value"])


dates = dt.datetime.date(dt.datetime.strptime(date_list[1], "%Y-%m-%d"))

datelist = pd.date_range(pd.datetime.today(), periods = number).tolist()

for x in range (0,number):
    tempdate = dt.datetime.strptime(date_list[x+1], "%Y-%m-%d")
    datelist[x] =dt.datetime.date(tempdate)



fig = plt.figure()
fig.suptitle(series, fontsize=14, fontweight='bold')

plt.plot(datelist, gdp_array)
plt.gcf().autofmt_xdate()
plt.show()

