# NCAA March Madness Prediction Model

## Project Overview

I created this project to see if data science could outperform traditional prediction methods. By leveraging machine learning, I aimed to build a model that accurately predicts NCAA March Madness game outcomes.  

I collected and processed 8,939 games from ESPN, tested multiple machine learning models, and evaluated their performance to determine the most effective approach.  

---

## Phase 1: Collecting the Raw Data  

Building a predictive model starts with gathering high-quality data. I scraped **8,939 NCAA basketball games** from ESPN, collecting key information such as:  

- Final scores and point differentials  
- Home/away team designations  
- Detailed box score statistics  
- Game dates and team matchups  

### Web Scraping Process  

To collect this data, I used **BeautifulSoup**, **Selenium**, and **pandas** to scrape ESPN’s website efficiently.  

```python
import datetime
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Set up headless Chrome browser
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

# Initialize DataFrame to store all game data
gameStats = pd.DataFrame()

# Set date range (Oct 6, 2023 - Feb 20, 2024)
start_date = datetime.date(2023, 10, 6)
end_date = datetime.date(2024, 2, 20)
delta = datetime.timedelta(days=1)

# Main scraping loop
while start_date <= end_date:
    print(f"Scraping {start_date.strftime('%Y-%m-%d')}...")
    
    # Get scoreboard page
    date_str = start_date.strftime("%Y%m%d")
    url = f'https://www.espn.com/mens-college-basketball/scoreboard/_/date/{date_str}'
    driver.get(url)
    
    # Parse game links
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    game_links = soup.find_all('a', string='Box Score')
    
    # Process each game
    for link in game_links:
        try:
            game_tables = pd.read_html('http://espn.com' + link['href'])
            home_team = game_tables[0].iloc[1, 0]
            away_team = game_tables[0].iloc[0, 0]
            home_score = game_tables[0].iloc[1, 3]
            away_score = game_tables[0].iloc[0, 3]
            result = 1 if home_score > away_score else 0
            
            game_data = pd.DataFrame([[start_date, home_team, away_team, home_score, away_score, result]],
                                     columns=['Date', 'HomeTeam', 'AwayTeam', 'HomeScore', 'AwayScore', 'Result'])
            gameStats = pd.concat([gameStats, game_data], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing game: {e}")
            continue
    
    start_date += delta

gameStats.to_csv('ncaa_basketball_games_2023_24.csv')
print(f"Successfully scraped {len(gameStats)} games!")

```

## Phase 2: Cleaning and Organizing the Data  

After retrieving the dataset, I cleaned and processed the data to ensure accuracy.  

- **Rolling Averages:** I calculated rolling averages using Python but encountered an issue when computing the rolling average for a team’s first three games. This led to the inclusion of data from previous teams. I resolved this by implementing an **IF statement in Excel** to ensure correct calculations.  
- **Win Tracking:** I computed the running total of team wins to capture momentum and performance trends.  

---

## Phase 3: Structuring the Dataset  

With a clean dataset, I structured it to include relevant features for predicting game outcomes. The final dataset contained:  

- **Date:** When the game was played  
- **Team:** Which team played  
- **Opp:** Opponent team  
- **PTS:** Points scored by the home team  
- **OPPpts:** Points scored by the away team  
- **Home:** Whether the game was played at home  
- **PD:** Point differential (team score minus opponent score)  
- **Win:** Whether the team won (1) or lost (0)  
- **rowcount:** Total games played in the timeframe  
- **Total Wins:** Cumulative wins for the team  
- **AvgPD:** Average Point Differential  

![Dataset Preview](https://github.com/user-attachments/assets/87d041ba-853f-4eda-a4ba-15f19eef7767)  

---

## Phase 4: Model Selection and Analysis  

I applied three machine learning techniques to predict game outcomes:  

1. **Logistic Regression** – A simple, interpretable classification model.  
2. **Decision Tree** – Captures non-linear relationships and visualizes decision-making.  
3. **Random Forest** – Reduces overfitting by combining multiple decision trees for more stable predictions.  

