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

# 🏀 NCAA March Madness Prediction Model

## 📘 Project Overview
This project explores whether data science can outperform traditional prediction methods in NCAA March Madness. By leveraging machine learning, the goal is to accurately predict game outcomes using statistical insights and modeling techniques.

I collected and processed **8,939 NCAA basketball games** from ESPN, evaluated multiple machine learning models, and selected the most effective for bracket prediction.

---

## 📊 Phase 1: Data Collection
Using Python libraries like `BeautifulSoup`, `Selenium`, and `pandas`, I scraped over 8,900 games from ESPN, capturing:
- Final scores and point differentials
- Home/away designations
- Game dates and team matchups
- Box score statistics

A headless browser was used to automate navigation and extract data efficiently across a set date range.

---

## 🧼 Phase 2: Data Cleaning
To ensure model accuracy, I cleaned and validated the dataset:
- Fixed inconsistent team naming
- Addressed missing values and outliers
- Calculated rolling averages (handled edge cases using Excel logic)
- Tracked win momentum for each team

---

## 🧱 Phase 3: Dataset Structuring
The dataset was structured to include:
- `Date`: Game date
- `Team` and `Opp`: Matchup info
- `PTS` and `OPPpts`: Points scored
- `Home`: Home court flag
- `PD`: Point Differential
- `Win`: Binary win/loss indicator
- `rowcount`: Number of games played
- `TotalWins`: Running win total
- `AvgPD`: Average point differential over recent games

![Dataset Preview](https://github.com/user-attachments/assets/87d041ba-853f-4eda-a4ba-15f19eef7767)

---

## 🧠 Phase 4: Model Selection
I trained three ML models to classify game outcomes:
1. **Logistic Regression** – Interpretable baseline model
2. **Decision Tree** – Captures non-linear features
3. **Random Forest** – Robust ensemble approach

Performance was evaluated using accuracy and AUC.

---

## 💡 Bonus: Dream Team Momentum Analysis
To enhance prediction accuracy, I added a "Dream Team" feature that highlights team momentum leading up to each game. This script uses a rolling 3-game window to compute:

- 🏆 `TotalWins`: Number of wins over the last 3 games
- 📊 `AvgPD`: Average Point Differential (team score minus opponent score)

These rolling metrics help identify hot teams and can be used as additional model inputs.

### 🔍 Sample Code
```python
import pandas as pd

df = pd.read_excel('gameStats.xlsx')

homeDF = df[['Date','HomeTeam','AwayTeam','homePTS','awayPTS']].copy()
awayDF = df[['Date','AwayTeam','HomeTeam','awayPTS','homePTS']].copy()

homeDF.rename(columns={"HomeTeam":"Team","AwayTeam":"Opp","homePTS":"PTS","awayPTS":"OPPpts"}, inplace=True)
awayDF.rename(columns={"AwayTeam":"Team","HomeTeam":"Opp","awayPTS":"PTS","homePTS":"OPPpts"}, inplace=True)

homeDF['Home'] = 1
awayDF['Home'] = 0

allGames = pd.concat([homeDF, awayDF])
allGames['PD'] = allGames['PTS'] - allGames['OPPpts']
allGames['Win'] = (allGames['PD'] >= 0).astype(int)
allGames.sort_values(by=['Team', 'Date'], inplace=True)
allGames['rowcount'] = allGames.groupby('Team').cumcount() + 1

rollingWindow = 3
allGames['TotalWins'] = allGames.groupby('Team')['Win'].rolling(rollingWindow, closed='left').sum().reset_index(0, drop=True)
allGames['AvgPD'] = allGames.groupby('Team')['PD'].rolling(rollingWindow, closed='left').mean().reset_index(0, drop=True)

dream_team_stats = allGames[allGames['rowcount'] > rollingWindow]
```

---

## 🧩 Pseudocode Overview
```python
# Step 1: Load Dataset
# Step 2: Clean data (nulls, inconsistencies, categorization)
# Step 3: Select relevant features
# Step 4: Create new structured dataset
# Step 5: Define target variable (Result)
# Step 6: Split data into training/testing sets
# Step 7: Preprocess data
# Step 8: Create ML models (Logistic Regression, Random Forest, Decision Tree)
# Step 9: Train models
# Step 10: Evaluate models (AUC, Confusion Matrix)
# Step 11: Apply model to make predictions
```

---

## 📬 Contact
Built by Abdullah Subuh | [LinkedIn](https://www.linkedin.com/in/asubuh111)

