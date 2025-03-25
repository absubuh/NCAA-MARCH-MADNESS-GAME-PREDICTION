#   NCAA March Madness Prediction Model

##   Project Overview

   Every year, millions of basketball fans try to create the perfect March Madness bracket. The odds are astronomically against them - the chance of predicting all 63 tournament games correctly is about 1 in 9.2 quintillion. Even predicting just the Final Four teams perfectly happens less than 1% of the time. I created this project to see if data science could do better than gut feelings and office pool strategies.

##   Data Acquisition

###   Getting the Data with Python

   I got the data I needed by scraping it from the ESPN website using Python. Here's how I did it:

   * **Choosing where to get the data:** I picked ESPN because they have a ton of detailed info on college basketball games.
   * **Grabbing the data:** I used Python code to automatically go through the website and pull out the game data I wanted.

The data I used included:

   * `Date`: When the game was played.
   * `Team`: Which team played.
   * `Opponent`: Who they played against.
   * `HomeTeam`: Name of home team
   * `AwayTeam`: Name of away team
   * `Points (Home Team)`: Points scored by the home team.
   * `Opponent Points`: Points scored by the away team.
   * `Home`: If the game was at home.
   * `PD`: How many more points one team scored than the other.
   * `Win`: If the team won (1) or lost (0).
   * `rowcount`: How many games the team played in our timeframe.
   * `Total Wins`: Total wins for the team
   * `AvgPD`: Average Point Differential

   **![Image](https://github.com/user-attachments/assets/d4988b20-403f-4fef-94ea-116b23252ad6)**

## Phase 1: Collecting the Raw Materials
Building a prediction model starts with gathering the right ingredients. I scraped **8,939 NCAA basketball games** from ESPN, collecting:

- Final scores and point differentials
- Home/away team designations
- Detailed box score statistics
- Game dates and team matchups

<details>
<summary>📊 Click to view the ESPN scraping code (collected 8,939 games)</summary>

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
            # Extract game data
            game_tables = pd.read_html('http://espn.com' + link['href'])
            
            # Get teams and scores
            home_team = game_tables[0].iloc[1, 0]
            away_team = game_tables[0].iloc[0, 0]
            home_score = game_tables[0].iloc[1, 3]
            away_score = game_tables[0].iloc[0, 3]
            result = 1 if home_score > away_score else 0
            
            # Process box score data
            home_stats = process_box_score(game_tables[4], 'home')
            away_stats = process_box_score(game_tables[2], 'away')
            
            # Combine all data
            game_data = pd.concat([
                pd.DataFrame([[start_date, home_team, away_team, home_score, away_score, result]],
                             columns=['Date','HomeTeam','AwayTeam','HomeScore','AwayScore','Result']),
                home_stats,
                away_stats
            ], axis=1)
            
            gameStats = pd.concat([gameStats, game_data], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing game: {e}")
            continue
    
    start_date += delta

# Save final dataset (8,939 games)
gameStats.to_csv('ncaa_basketball_games_2023_24.csv')
print(f"Successfully scraped {len(gameStats)} games!")
```
</details>

## Phase 2: Cleaning and Organizing the Data
Raw sports data is like a messy locker room - everything's there, but not where you need it. I:

1. Fixed calculation errors in rolling averages
2. Standardized team names (e.g., "UNC" vs "North Carolina")
3. Created consistent date formats
4. Handled missing data points

The most valuable lesson came when I discovered my rolling averages were accidentally "peeking" at future games. Fixing this data leakage problem was crucial.

[INSERT BEFORE/AFTER DATA CLEANING COMPARISON]

## Phase 3: Discovering the Hidden Patterns
Before building models, I explored the data visually. Some key insights:

- Teams with consistent +5 point margins won 78% of subsequent games
- Home court advantage added just 3.2 points on average
- Recent performance (last 3 games) mattered more than season-long stats

[INSERT INFOGRAPHIC OF KEY FINDINGS]

## Phase 4: Building Prediction Models
I tested three approaches, each with different strengths:

### 1. The Straightforward Scout (Logistic Regression)
- 72% accurate
- Easy to interpret
- Missed some subtle patterns

[INSERT LOGISTIC REGRESSION RESULTS]

### 2. The Team of Experts (Random Forest)
- 78% accurate
- Captured complex relationships
- Harder to explain

[INSERT RANDOM FOREST FEATURE IMPORTANCE]

### 3. The Decision Tree Playbook
- 75% accurate
- Clear visual explanations
- Sometimes oversimplified

[INSERT DECISION TREE VISUALIZATION]

## The Hard Truth About "Perfect" Predictions
My initial models claimed 100% accuracy - an obvious red flag. Discovering and fixing this taught me more than any textbook about:

- The importance of proper validation
- How easily time-series data can leak
- Why skepticism is a data scientist's best tool

[INSERT MODEL ACCURACY COMPARISON CHART]
