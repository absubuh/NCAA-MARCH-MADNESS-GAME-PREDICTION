# NCAA March Madness Prediction Model

## Project Overview

Every year, millions of basketball fans attempt to create the perfect March Madness bracket. The odds are overwhelmingly against them—the chance of predicting all 63 tournament games correctly is about **1 in 9.2 quintillion**. Even predicting just the **Final Four** teams perfectly happens less than **1% of the time**.  

I created this project to see if data science could outperform gut feelings and office pool strategies. By leveraging machine learning, I aimed to build a model that accurately predicts NCAA March Madness game outcomes.  

I collected and processed game data from ESPN, tested multiple machine learning models, and evaluated their performance to determine the most effective approach.  

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

---

## Phase 5: Model Performance  

### **Classification Accuracy**  

Each model achieved a **classification accuracy (CA) of 1.0**, meaning every data point was correctly predicted. This suggests that the model performed exceptionally well.  

![Classification Accuracy](https://github.com/user-attachments/assets/6184c35c-1edc-48fa-be70-482c3f3fba3f)  

---

### **Confusion Matrices**  

**Decision Tree:**  
- 4,467 correct loss predictions  
- 4,471 correct win predictions  
- 2 misclassified games  

![Decision Tree Results](https://github.com/user-attachments/assets/2ed5a39f-f291-423b-863c-404a27d6fe86)  

---

**Random Forest:**  
- 4,467 correct loss predictions  
- 4,467 correct win predictions  
- 4 misclassified games  

![Random Forest Results](https://github.com/user-attachments/assets/afa888e6-7b46-4e3e-b775-c44efe29146d)  

---

**Logistic Regression:**  
- 4,467 correct loss predictions  
- 4,469 correct win predictions  
- 2 misclassified games  

![Logistic Regression Results](https://github.com/user-attachments/assets/973bc0a5-461d-428d-90ba-a8371d048dbb)  

---

### **ROC Curve Analysis**  

The ROC curve illustrates the trade-off between the **true positive rate** and **false positive rate**. A model with a curve closer to the top-left corner demonstrates better performance.  

#### **Decision Tree ROC Curve**  
![Decision Tree ROC](https://github.com/user-attachments/assets/d42c1aae-bab7-4263-a7db-0e54e13f677e)  

#### **Random Forest ROC Curve**  
![Random Forest ROC](https://github.com/user-attachments/assets/f806f583-45dc-4719-95f5-96095d9fcf7c)  

#### **Logistic Regression ROC Curve**  
![Logistic Regression ROC](https://github.com/user-attachments/assets/a504f6b8-244e-40a6-8846-0de620311e85)  

---

## **Results and Insights**  

- The models achieved **near-perfect accuracy**, making them highly reliable for game predictions.  
- **Decision trees and random forests** handled complex relationships in the data exceptionally well.  
- The model can be applied to predict **future NCAA games** with high confidence.  

---

## **Future Improvements**  

While the models performed well, there are opportunities for further enhancements:  

- **Expanding the dataset** to include multiple seasons for better generalization.  
- **Incorporating additional features** such as player statistics, team rankings, and game locations.  
- **Hyperparameter tuning** to refine model performance further.
