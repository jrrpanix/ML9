import itertools
import urllib.request
from bs4 import BeautifulSoup
import re
import pandas as pd
import os

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
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20180131.htm",
  "https://www.federalreserve.gov/monetarypolicy/fomcminutes20190130.htm"]


listOfStmtURLs = {
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
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20180131a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20190320a.htm",
  "https://www.federalreserve.gov/newsevents/pressreleases/monetary20190130a.htm"
}

response =  urllib.request.urlopen('https://www.federalreserve.gov/newsevents/pressreleases/monetary20140129a.htm')
html = response.read()
soup = BeautifulSoup(html,'html5lib')
text2 = soup.get_text(strip = True)
text2= text2[text2.find('For immediate releaseShare')+26:]
text2=text2[:text2.index("Last Update:")+100]


##Minute Text Retrieval
for x in listOfMinutesURLs:
  response = urllib.request.urlopen(x)
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
  if text2.rfind(end) == False:
    print("no time")
  else: 
    meeting_date = (text2[text2.find(start)+len(start):text2.rfind(end)-17]).strip()
    if len(meeting_date) > 75: 
      print("problem")
    else: 
      print(meeting_date)
  print("\n")

##Statement Text Retrieval
for x in listOfStmtURLs:
  response = urllib.request.urlopen(x)
  html = response.read()
  soup = BeautifulSoup(html,'html5lib')
  text2 = soup.get_text(strip = True)
  text2= text2[text2.find("Share"):]
  text2=text2[:text2.index("Last Update:")+100]
  start = 'Last Update:'
  end = 'Board of Gov'
  meeting_date = text2[text2.find(start)+len(start):text2.rfind(end)].strip()
  print(meeting_date)

  

#get a list of URLS of Governors speeches for a given year
#Scrape the dates and names from the website for a given year

listOfYears = ['2014','2015','2016','2017','2018','2019']

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
  search_list = ['yellen', 'powell', 'fischer','tarullo','quarles','brainard','clarida','stein']
  long_string = text2
  names_reg_Ex = re.compile('|'.join(search_list),re.IGNORECASE) #re.IGNORECASE is used to ignore case
  name_matches_list = names_reg_Ex.findall(text2)

  #Combine the two
  for index, match in enumerate(date_matches_list):
    date_matches_list[index] = datetime.datetime.strptime(match, "%m/%d/%Y").strftime("%Y%m%d")
    urls.append('https://www.federalreserve.gov/newsevents/speech/'+name_matches_list[index].lower()+date_matches_list[index]+'a.htm')
    #date_matches_list[index] = datetime.datetime.strptime(match, "%m/%d/%Y").strftime("%Y%m%d") 

  #Get Text from website, save to a folder
  for index, item in enumerate(urls):
    response = urllib.request.urlopen(item)
    html = response.read()
    soup = BeautifulSoup(html,'html5lib')
    text2 = soup.get_text(strip = True)
    text2= text2[text2.find("Share"):]
    text2=text2[:text2.index("Last Update:")]
    text_file = open(os.getcwd()+"/text/speeches/"+date_matches_list[index]+"_"+name_matches_list[index].lower()+".txt","w")
    text_file.write(text2)
    text_file.close()
    print(index)


###Scratch Code:
##Retrieve the Date published
#start = 'Last Update:'
#end = 'Board of Gov'
#print(text2[text2.find(start)+len(start):text2.rfind(end)])

##Retrieve the meeting date
#start = 'adjourned at'
#end = 'notation vote'
#print(text2[text2.find(start)+len(start):text2.rfind(end)-17])
 
#text2[:text2.rfind(end)]
  
#import PyPDF2
#pdf_file = open('~/ML/project/FOMCpresconf20141217.pdf', 'rb')
#read_pdf = PyPDF2.PdfFileReader(pdf_file)
#number_of_pages = read_pdf.getNumPages()
#page = read_pdf.getPage(0)
#page_content = page.extractText()
#print (page_content.encode('utf-8'))




##Getting speech text

#response =  urllib.request.urlopen('https://www.federalreserve.gov/newsevents/speech/2017-speeches.htm')
#html = response.read()
#soup = BeautifulSoup(html,'html5lib')
#text2 = soup.get_text(strip = True)
#date_reg_exp = re.compile('\d{2}/\d{2}/\d{4}')
#date_matches_list=date_reg_exp.findall(text2)
#for dateMatch in date_matches_list:
#  print(dateMatch)

#search_list = ['yellen', 'powell', 'fischer','tarullo','quarles','brainard']
#long_string = text2
#names_reg_Ex = re.compile('|'.join(search_list),re.IGNORECASE) #re.IGNORECASE is used to ignore case
#name_matches_list = names_reg_Ex.findall(text2)
#for nameMatch in name_matches_list:
#  print(nameMatch)
  
#for index, match in enumerate(date_matches_list):
#  name_matches_list[index] = 'https://www.federalreserve.gov/newsevents/speech/'+name_matches_list[index].lower()+datetime.datetime.strptime(match, "%m/%d/%Y").strftime("%Y%m%d")+'a.htm'
#  #date_matches_list[index] = datetime.datetime.strptime(match, "%m/%d/%Y").strftime("%Y%m%d") 

#for match in name_matches_list:
#  print(match)

  

#for item in itertools.chain(date_matches_list, name_matches_list):
#  print(item)
