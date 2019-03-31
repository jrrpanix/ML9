import itertools
import urllib
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import datetime

listOfMinutesURLs = [
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20141217.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20141029.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20140917.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20140730.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20140618.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20140430.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20140319.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20140129.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20151216.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20151028.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20150917.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20150729.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20150617.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20150429.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20150318.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20150128.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20161214.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20161102.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20160921.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20160727.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20160615.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20160427.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20160316.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20160127.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20171213.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20171101.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20170920.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20170726.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20170614.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20170503.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20170315.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20170201.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20181219.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20181108.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20180926.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20180801.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20180613.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20180502.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20180321.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20180131.htm"
  ]


listOfStmtURLs = [
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20141217a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20141029a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140917a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140730a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140618a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140430a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140319a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140129a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20151216a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20151028a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20150917a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20150729a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20150617a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20150429a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20150318a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20150128a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20161214a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20161102a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20160921a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20160727a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20160615a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20160427a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20160316a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20160127a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20171213a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20171101a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20170920a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20170726a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20170614a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20170503a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20170315a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20170201a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20181219a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20181108a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180926a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180801a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180613a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180502a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180321a.htm"
]

#Change relative directory
os.chdir("..")
os.chdir(os.path.abspath(os.curdir)+"/text")


##Minute Text Retrieval
for index,x in enumerate(listOfMinutesURLs):
  response = urllib.request.urlopen(listOfMinutesURLs[index])
  html = response.read()
  soup = BeautifulSoup(html,'html5lib')
  text2 = soup.get_text(strip = True)
  text2= text2[text2.find("Developments in Financial Markets"):]
  text2=text2[:text2.index("Last Update")+100]
  start = 'Last Update:'
  end = 'Board of Gov'
  publish_date = (text2[text2.find(start)+len(start):text2.rfind(end)]).strip()
  print(publish_date)
  start = 'adjourned at'
  end = 'notation vote'
  prog = re.compile('\d{4}\d{2}\d{2}')
  dateOfText=re.findall(prog,listOfMinutesURLs[index])
  publishDate = datetime.datetime.strptime(publish_date,"%B %d, %Y").strftime("%Y%m%d")
  text_file = open(os.getcwd()+"/minutes/"+dateOfText[0]+"_minutes_published_"+publishDate+".txt","w")
  text_file.write(text2)
  text_file.close()

##Statement Text Retrieval
for index,x in enumerate(listOfStmtURLs):
  response = urllib.request.urlopen(listOfStmtURLs[index])
  html = response.read()
  soup = BeautifulSoup(html,'html5lib')
  text2 = soup.get_text(strip = True)
  text2= text2[text2.find("Share"):]
  text2=text2[:text2.index("Last Update:")+100]
  start = 'Last Update:'
  end = 'Board of Gov'
  meeting_date = text2[text2.find(start)+len(start):text2.rfind(end)].strip()
  print(meeting_date)
  start = 'adjourned at'
  end = 'notation vote'
  prog = re.compile('\d{4}\d{2}\d{2}')
  dateOfText=re.findall(prog,listOfStmtURLs[index])
  text_file = open(os.getcwd()+"/statements/"+dateOfText[0]+".txt","w")
  text_file.write(text2)
  text_file.close()

  

#get a list of URLS of Governors speeches for a given year
#Scrape the dates and names from the website for a given year

listOfYears = ['2014','2015','2016']

#2014 and 2015 need to be evaluated manually 
search_list = ['yellen', 'powell', 'fischer','tarullo','quarles','brainard','clarida','stein']



for year in listOfYears:
  #Build list of dates
  urls = list()
  speechYearUrl = "https://www.federalreserve.gov/newsevents/speech/"+year+"-speeches.htm"
  response =  urllib.request.urlopen(speechYearUrl)
  html = response.read()
  soup = BeautifulSoup(html,'html5lib')
  text2 = soup.get_text(strip = True)
  date_reg_exp = re.compile('\d{2}/\d{2}/\d{4}')
  date_matches_list=date_reg_exp.findall(text2)
  #Build list of names
  long_string = text2
  names_reg_Ex = re.compile('|'.join(search_list),re.IGNORECASE) #re.IGNORECASE is used to ignore case
  name_matches_list = names_reg_Ex.findall(text2)
  #Combine the two
  for idex, match in enumerate(date_matches_list):
    for jdex, speaker in enumerate(search_list):
      try:
        date_matches_list[idex] = datetime.datetime.strptime(match, "%m/%d/%Y").strftime("%Y%m%d")
        urlCall = 'https://www.federalreserve.gov/newsevents/speech/'+search_list[jdex].lower()+date_matches_list[idex]+'a.htm'
        print(urlCall)
        response = urllib.request.urlopen(urlCall)
        html = response.read()
        soup = BeautifulSoup(html,'html5lib')
        text2 = soup.get_text(strip = True)
        text2= text2[text2.find("Share"):]
        text2=text2[:text2.index("Last Update:")]
        text_file = open(os.getcwd()+"/text/speeches/"+date_matches_list[idex]+"_"+name_matches_list[idex].lower()+".txt","w")
        text_file.write(text2)
        text_file.close()
        print(idex)
        break
      except HTTPError as e:
        print('Error code: ', e.code)
      except URLError as e:
        print('Reason: ', e.reason)

