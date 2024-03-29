# -*- coding: utf-8 -*-
"""ESPNscrape.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NZdW9EMekDSb6zg_UuR9aznIBBkpStQe
"""

import datetime

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install webdriver-manager
# !pip install selenium

from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

import datetime

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('start-maximized')
user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'
chrome_options.add_argument('user-agent={0}'.format(user_agent))

#s=Service('/usr/bin/chromedriver')
driver=webdriver.Chrome(options=chrome_options)

gameStats=pd.DataFrame()

start_date = datetime.date(2023, 10, 6)
end_date = datetime.date(2024, 2, 20)

# delta time
delta = datetime.timedelta(days=1)

# iterate over range of dates

while (start_date <= end_date):
  print(start_date,datetime.datetime.now())
  dateString=start_date.strftime("%Y%m%d")

  #Page containing all games played for a specific day
  driver.get('https://www.espn.com/mens-college-basketball/scoreboard/_/date/' + dateString + '/seasontype/2/group/50')
  scorePage=driver.execute_script('return document.body.innerHTML')
  parsedScorePage=BeautifulSoup(scorePage,'html.parser')

  #Get all game links
  gameLinks=parsedScorePage.find_all('a',string='Box Score')

  #loop through this day's games
  for link in gameLinks:
    #get all tables from the game page
    gameTables=pd.read_html('http://espn.com' + link['href'])

    #get the names of the home and away teams
    homeTeam=gameTables[0].iloc[1,0]
    awayTeam=gameTables[0].iloc[0,0]
    homeScore=gameTables[0].iloc[1,3]
    awayScore=gameTables[0].iloc[0,3]
    if homeScore > awayScore:
      result=1
    else:
      result=0

    #create a table for the home team
    homeData=gameTables[4].iloc[[-2]]
    colnames=gameTables[4].iloc[[0]].values.tolist()[0]
    for i in range(len(colnames)):
      colnames[i]='home'+colnames[i]
    homeData.columns=colnames

    #create a table for the away team
    awayData=gameTables[2].iloc[[-2]]
    colnames=gameTables[2].iloc[[0]].values.tolist()[0]
    for i in range(len(colnames)):
      colnames[i]='away'+colnames[i]
    awayData.columns=colnames

    gameData=pd.concat([homeData.reset_index(),awayData.reset_index()], axis=1)

    #create a mini table containing the names of the home and away teams
    teamNameDF=pd.DataFrame(columns=['Date','HomeTeam','AwayTeam','HomeScore','AwayScore','Result'],data=[[start_date,homeTeam,awayTeam,homeScore,awayScore,result]])
    miniFrame=pd.concat([teamNameDF.reset_index(),gameData.reset_index()],axis=1)
    miniFrame.drop(columns=['index'],inplace=True)

    gameStats=pd.concat([gameStats,miniFrame],ignore_index=True)
  start_date += delta

gameStats.to_csv('/content/drive/My Drive/Colab Notebooks/FinalFour/gameStats.csv')

gameStats