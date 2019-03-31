import os
import pandas


def statementsToPandas():
  os.chdir(os.path.abspath(os.curdir)+"/statements")
  for file in os.listdir(os.getcwd()):
    print(file)
    if file.endswith(".txt"):
      with open(file,"r") as f:
        new = ""
        for line in f: 
          new = line.strip() + new 
  print(new)
  os.chdir("..")


def main():
  #Change relative directory
  os.chdir("..")
  os.chdir(os.path.abspath(os.curdir)+"/text")

  
if __name__ == '__main__':
  main()
  statementsToPandas()

#pandas.DataFrame({'text': text,
#                  'dataRelease' : dateRelease,
#                   'docType' : docType,
#                   'meetingResult' : meetingResult},
#                   index = ['1'])
                   
                   
#pandas.DataFrame({'text': text}, index = ['1'])