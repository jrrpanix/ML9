import itertools
import urllib
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import datetime

def retrieveMinutes():
  listOfMinutesURLs = [
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20120125.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20120313.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20120425.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20120620.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20120801.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20120913.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20121024.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20121212.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20130130.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20130320.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20130501.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20130619.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20130731.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20130918.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20131030.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20131218.htm",
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
  
  ##Minute Text Retrieval
  for index,x in enumerate(listOfMinutesURLs):
    response = urllib.request.urlopen(listOfMinutesURLs[index])
    html = response.read()
    soup = BeautifulSoup(html,'html5lib')
    text2 = soup.get_text(strip = True)
    text2= text2[text2.find("in Financial Markets"):]
    text2=text2[:text2.index("Last Update")+100]
    start = 'Last Update:'
    end = 'Board of Gov'
    publish_date = (text2[text2.find(start)+len(start):text2.rfind(end)]).strip()
    print(publish_date)
    text2=re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text2)
    text2 = text2[:text2.index("Notation")]
    prog = re.compile('\d{4}\d{2}\d{2}')
    dateOfText=re.findall(prog,listOfMinutesURLs[index])
    publishDate = datetime.datetime.strptime(publish_date,"%B %d, %Y").strftime("%Y%m%d")
    text_file = open(os.getcwd()+"/minutes/"+dateOfText[0]+"_minutes_published_"+publishDate+".txt","w")
    text_file.write(text2)
    text_file.close()


def retrieveOldWebsiteMinutes():
  listOfMinutesURLs = [
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20100127.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20100316.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20100428.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20100623.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20100810.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20100921.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20101103.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20101214.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20110126.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20110315.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20110427.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20110622.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20110809.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20110921.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20111102.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20111213.htm"
  ]
  
  ##Minute Text Retrieval
  for index,x in enumerate(listOfMinutesURLs):
    print (listOfMinutesURLs[index])
    response = urllib.request.urlopen(listOfMinutesURLs[index])
    html = response.read()
    soup = BeautifulSoup(html,'html5lib')
    text2 = soup.get_text(strip = True)
    text2=re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text2)
    text2= text2[text2.index("Developments in Financial Markets"):]
    text2=text2[:text2.index("Last update")+100]
    start = 'Last update:'
    end = 'Home'
    publish_date = (text2[text2.find(start)+len(start):text2.rfind(end)]).strip()
    print(publish_date)
    if text2.find("Notation") == -1:
      text2 = text2[:text2.index("Return to top")]
    else: 
      text2 = text2[:text2.index("Notation")]
    prog = re.compile('\d{4}\d{2}\d{2}')
    dateOfText=re.findall(prog,listOfMinutesURLs[index])
    publishDate = datetime.datetime.strptime(publish_date,"%B %d, %Y").strftime("%Y%m%d")
    text_file = open(os.getcwd()+"/minutes/"+dateOfText[0]+"_minutes_published_"+publishDate+".txt","w")
    text_file.write(text2)
    text_file.close()

    
def retrieveStatements():
  listOfStmtURLs = [
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100127a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100316a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100428a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100623a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100810a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100921a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20101103a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20101214a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20110126a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20110315a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20110427a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20110622a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20110809a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20110921a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20111102a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20111213a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120125a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120313a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120425a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120620a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120801a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120913a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20121024a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20121212a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20130130a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20130320a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20130501a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20130619a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20130731a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20130918a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20131030a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20131218a.htm",
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
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180321a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180131a.htm"
  ]
  
  ##Statement Text Retrieval
  for index,x in enumerate(listOfStmtURLs):
    response = urllib.request.urlopen(listOfStmtURLs[index])
    html = response.read()
    soup = BeautifulSoup(html,'html5lib')
    text2 = soup.get_text(strip = True)
    text2= text2[text2.find("Information received since"):]
    text2=text2[:text2.index("Last Update:")]
    text2=re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text2)
    prog = re.compile('\d{4}\d{2}\d{2}')
    dateOfText=re.findall(prog,listOfStmtURLs[index])
    text_file = open(os.getcwd()+"/statements/"+dateOfText[0]+".txt","w")
    text_file.write(text2)
    text_file.close()

def retrieveSpeeches():
  #get a list of URLS of Governors speeches for a given year
  #Scrape the dates and names from the website for a given year
  listOfYears = ['2011','2012','2013','2014','2015','2016','2017','2018']
  search_list = ['yellen', 'powell', 'fischer','tarullo','quarles','brainard','clarida','stein','bernanke','duke','raskin']
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
    long_string = text2
    names_reg_Ex = re.compile('|'.join(search_list),re.IGNORECASE) #re.IGNORECASE is used to ignore case
    name_matches_list = names_reg_Ex.findall(text2)
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
          text2= text2[text2.find("Please enable JavaScript if it is disabled in your browser or access the information through the links provided below"):]
          text2= text2[text2.find("Share"):]
          text2=text2[:text2.index("Last Update:")]
          text2 = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text2)
          if text2.find("1.") != -1:
            text2=text2[:text2.find("1.")]
          text_file = open(os.getcwd()+"/speeches/"+date_matches_list[idex]+"_"+name_matches_list[idex].lower()+".txt","w")
          text_file.write(text2)
          text_file.close()
          print(idex)
          break
        except HTTPError as e:
          print('Error code: ', e.code)
        except URLError as e:
          print('Reason: ', e.reason)

def retrieveMinutesWOStatement():
  listOfMinutesURLs = [
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20120125.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20120313.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20120425.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20120620.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20120801.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20120913.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20121024.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20121212.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20130130.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20130320.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20130501.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20130619.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20130731.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20130918.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20131030.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20131218.htm",
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
  
  ##Minute Text Retrieval
  for index,x in enumerate(listOfMinutesURLs):
    response = urllib.request.urlopen(listOfMinutesURLs[index])
    html = response.read()
    soup = BeautifulSoup(html,'html5lib')
    text2 = soup.get_text(strip = True)
    text2= text2[text2.find("in Financial Markets"):]
    text2=text2[:text2.index("Last Update")+100]
    start = 'Last Update:'
    end = 'Board of Gov'
    publish_date = (text2[text2.find(start)+len(start):text2.rfind(end)]).strip()
    print(publish_date)
    text2=re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text2)
    text2 = text2[:text2.rfind("At the conclusion of the discussion")]
    prog = re.compile('\d{4}\d{2}\d{2}')
    dateOfText=re.findall(prog,listOfMinutesURLs[index])
    publishDate = datetime.datetime.strptime(publish_date,"%B %d, %Y").strftime("%Y%m%d")
    text_file = open(os.getcwd()+"/minutes_wo_statement/"+dateOfText[0]+"_minutes_published_"+publishDate+".txt","w")
    text_file.write(text2)
    text_file.close()


def retrieveOldWebsiteMinutesWOStatement():
  listOfMinutesURLs = [
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20100127.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20100316.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20100428.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20100623.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20100810.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20100921.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20101103.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20101214.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20110126.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20110315.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20110427.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20110622.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20110809.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20110921.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20111102.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20111213.htm"
  ]
  
  ##Minute Text Retrieval
  for index,x in enumerate(listOfMinutesURLs):
    print (listOfMinutesURLs[index])
    response = urllib.request.urlopen(listOfMinutesURLs[index])
    html = response.read()
    soup = BeautifulSoup(html,'html5lib')
    text2 = soup.get_text(strip = True)
    text2=re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text2)
    text2= text2[text2.index("Developments in Financial Markets"):]
    text2=text2[:text2.index("Last update")+100]
    start = 'Last update:'
    end = 'Home'
    publish_date = (text2[text2.find(start)+len(start):text2.rfind(end)]).strip()
    print(publish_date)
    if text2.rfind("At the conclusion of the discussion") == -1:
      text2 = text2[:text2.index("Return to top")]
    else: 
      text2 = text2[:text2.rfind("At the conclusion of the discussion")]
    prog = re.compile('\d{4}\d{2}\d{2}')
    dateOfText=re.findall(prog,listOfMinutesURLs[index])
    publishDate = datetime.datetime.strptime(publish_date,"%B %d, %Y").strftime("%Y%m%d")
    text_file = open(os.getcwd()+"/minutes_wo_statement/"+dateOfText[0]+"_minutes_published_"+publishDate+".txt","w")
    text_file.write(text2)
    text_file.close()

def main():
  #Change relative directory
  os.chdir("..")
  os.chdir(os.path.abspath(os.curdir)+"/text")
  retrieveStatements()
  retrieveMinutes()
  retrieveOldWebsiteMinutes()
  retrieveSpeeches()
  retrieveMinutesWOStatement()
  retrieveOldWebsiteMinutesWOStatement()
  
if __name__ == '__main__':
  main()




