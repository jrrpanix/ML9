import itertools
import urllib
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import datetime
import csv
from datetime import datetime
from dateutil.parser import parse

def pre08_MinutesScraper(meetingList):

  for i in meetingList:
    url = 'https://www.federalreserve.gov/fomc/minutes/'+i[0]+'.htm'
    secondUrl = 'https://www.federalreserve.gov/monetarypolicy/fomcminutes'+i[0]+'.htm'
    thirdUrl = 'https://www.federalreserve.gov/monetarypolicy/fomc20080625.htm'
    prog = re.compile('\d{4}\d{2}\d{2}')
    try:
      response = urllib.request.urlopen(url)
      dateOfText=re.findall(prog,url)
    except: 
      try:
        response = urllib.request.urlopen(secondUrl)
        dateOfText=re.findall(prog,secondUrl)
      except:
        response = urllib.request.urlopen(thirdUrl)
        dateOfText=re.findall(prog,thirdUrl)
      
    html = response.read()
    soup = BeautifulSoup(html,'html5lib')
    text2 = soup.get_text(strip = True)
    text2=re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text2)
    text2 = re.sub(r'\s+', ' ',text2).strip()
    startIndex = 0
    endIndex = 0
    if (text2.find("Developments in Financial Markets") != -1):
      startIndex = text2.index("Developments in Financial Markets")
    elif (text2.find("The information reviewed") != -1):
      startIndex = text2.index("The information reviewed")
    elif (text2.find("The information provided") != -1):
      startIndex = text2.index("The information provided")
    elif (text2.find("The Manager of the System Open Market") != -1):
      startIndex = text2.find("The Manager of the System Open Market")  
    else: 
      print(i[0])
      print("I SHOULDNT BE HERE")
    text2=text2[:text2.rfind("Last update")+100]
    start = ''
    end = ''
    publishDate = ''
    lengthText = len(text2)
    if(text2[lengthText-2:].lower() == 'pm'):
      start = 'Last update:'
      publish_date = (text2[text2.find(start)+len(start):]).replace(',','')
      publish_date = (publish_date).replace(' ','')
      publish_date_object = datetime.strptime(publish_date,'%B%d%Y%H:%M%p')
      publish_date = publish_date_object.strftime('%Y%m%d')
      publishDate = publish_date
    else: 
      start = 'Last update:'
      end = 'Home'
      publish_date = (text2[text2.find(start)+len(start):text2.rfind(end)]).strip()
      publishDate = datetime.strptime(publish_date,"%B %d, %Y").strftime("%Y%m%d")

    if text2.find("Notation") == -1:
      text2 = text2[:text2.rfind("Return to top")]
    else: 
      text2 = text2[:text2.rfind("Notation")]
    
    print(publish_date)
    text2 = re.sub("[^\x20-\x7E]", "",text2)
    text_file = open(os.getcwd()+"/minutes/"+dateOfText[0]+"_minutes_published_"+publishDate+".txt","w")
    print(dateOfText[0]+"_minutes_published_"+publishDate)
    text_file.write(text2)
    text_file.close()    

def retrieveMinutes():

  ##List of FOMC minutes URLs

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
    #Beautiful Soup to scrape the data from the URL
    response = urllib.request.urlopen(listOfMinutesURLs[index])
    html = response.read()
    soup = BeautifulSoup(html,'html5lib')
    text2 = soup.get_text(strip = True)
    
    #Index to the beginning and end of the minutes to remove unwanted javascript
    text2= text2[text2.find("in Financial Markets"):]
    text2=text2[:text2.index("Last Update")+100]
    start = 'Last Update:'
    end = 'Board of Gov'

    #Extract the minutes publish date
    publish_date = (text2[text2.find(start)+len(start):text2.rfind(end)]).strip()
    print(publish_date)

    #Add spaces between a lowercase and capital letter
    text2=re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text2)
    text2 = text2[:text2.rfind("At the conclusion of the discussion")]
    
    #Save file with date published and the date of the meeting
    prog = re.compile('\d{4}\d{2}\d{2}')
    dateOfText=re.findall(prog,listOfMinutesURLs[index])
    publishDate = datetime.strptime(publish_date,"%B %d, %Y").strftime("%Y%m%d")
    text_file = open(os.getcwd()+"/minutes/"+dateOfText[0]+"_minutes_published_"+publishDate+".txt","w")
    text_file.write(text2)
    text_file.close()


def retrieveOldWebsiteMinutes():
  listOfMinutesURLs = [
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20090128.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20090318.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20090429.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20090624.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20090812.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20090923.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20091104.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20091216.htm",
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
    
    #Older minutes are stored on an HTML website, no java script
    #Beautiful soup to scrape the minutes
    print (listOfMinutesURLs[index])
    response = urllib.request.urlopen(listOfMinutesURLs[index])
    html = response.read()
    soup = BeautifulSoup(html,'html5lib')
    text2 = soup.get_text(strip = True)
    text2=re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text2)

    #Capture the beginning and end of the minutes
    if(text2.find("Developments in Financial Markets") != -1):
      text2= text2[text2.index("Developments in Financial Markets"):]
      text2=text2[:text2.index("Last update")+100]
    else: 
      text2= text2[text2.index("Market Developments and Open Market Operations"):]
      text2=text2[:text2.index("Last update")+100]
      
    start = 'Last update:'
    end = 'Home'
    publish_date = (text2[text2.find(start)+len(start):text2.rfind(end)]).strip()
    print(publish_date)
    if text2.rfind("At the conclusion of the discussion") == -1:
      text2 = text2[:text2.index("Return to top")]
    else: 
      text2 = text2[:text2.rfind("At the conclusion of the discussion")]
    
    #Save minutes with the release and meeting dates in the file name
    prog = re.compile('\d{4}\d{2}\d{2}')
    dateOfText=re.findall(prog,listOfMinutesURLs[index])
    publishDate = datetime.strptime(publish_date,"%B %d, %Y").strftime("%Y%m%d")
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

  #The list of speakers over the 2011-2018 time period
  search_list = ['yellen', 'powell', 'fischer','tarullo','quarles','brainard','clarida','stein','bernanke','duke','raskin']

  for year in listOfYears:
    #Build list of dates
    urls = list()

    ##Speech urls are saved by year at this address, the date of the speech is represented as mm/dd/yyyy on this page, extract the dates from that page
    speechYearUrl = "https://www.federalreserve.gov/newsevents/speech/"+year+"-speeches.htm"
    response =  urllib.request.urlopen(speechYearUrl)
    html = response.read()
    soup = BeautifulSoup(html,'html5lib')#.decode('utf-8','ignore')
    text2 = soup.get_text(strip = True)
    date_reg_exp = re.compile('\d{2}/\d{2}/\d{4}')
    date_matches_list=date_reg_exp.findall(text2)
    long_string = text2
    names_reg_Ex = re.compile('|'.join(search_list),re.IGNORECASE) #re.IGNORECASE is used to ignore case
    
    #Iterate through the list of dates and possible speaker combinations, if there is a 404 error, the try/catch will try the next possible combination of date and speaker.  It will stop when it is able to extract a speech. 
    name_matches_list = names_reg_Ex.findall(text2)
    for idex, match in enumerate(date_matches_list):
      for jdex, speaker in enumerate(search_list):
        try:
          date_matches_list[idex] = datetime.strptime(match, "%m/%d/%Y").strftime("%Y%m%d")
          urlCall = 'https://www.federalreserve.gov/newsevents/speech/'+search_list[jdex].lower()+date_matches_list[idex]+'a.htm'
          print(urlCall)
          response = urllib.request.urlopen(urlCall)
          html = response.read()
          soup = BeautifulSoup(html,'html5lib')#.decode('utf-8','ignore')
          text2 = soup.get_text(strip = True)
 
          #Index to the start of the speech and end of the speech to exclude java script text
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


def main():
  #Change relative directory
  os.chdir("..")
  os.chdir(os.path.abspath(os.curdir)+"/text")
  retrieveStatements()
  retrieveMinutes()
  retrieveOldWebsiteMinutes()
  retrieveSpeeches()
  
  ##Get Pre 2008 Minutes
  fomcDate = []
  path = '../text/history/pre09MeetingList.csv'
  with open(path, 'r') as f:
    reader = csv.reader(f)
    fomcDate = list(reader)
  os.chdir("..")
  os.chdir(os.path.abspath(os.curdir)+"/text")
  pre08_MinutesScraper(fomcDate)  
if __name__ == '__main__':
  main()




