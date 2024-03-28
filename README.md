About My NCAA March Madness Tournament Prediction Model

For this project, our unit of analysis was predicting whether teams in the NCAA March Madness Tournament would win or lose their games. The goal was to understand team performance and predict outcomes for this year's tournament.

Data Collection and Preparation:
We web scraped data from the ESPN site using Python, focusing on games played from November 6th to February 10th. We collected data on teams, opponents, points scored, win/loss status, and other relevant metrics. Data cleaning involved calculating averages and running totals, with special attention to the first three games of each team to avoid data duplication.

Type of Analysis:
We used logistic regression, decision tree analysis, and random forest analysis for our model. Logistic regression provided straightforward conclusions, decision trees visualized nonlinear relationships, and random forests addressed potential limitations like overfitting.

Factors Included in the Analysis:
Our analysis included team performance metrics, win/loss records, average point differentials, and total wins. These factors were crucial for predicting game outcomes.

Results of the Analysis:

Classification Accuracy: All models achieved a classification accuracy of 1.0, indicating accurate predictions.
Confusion Matrix: The confusion matrices showed high accuracy, with few incorrectly predicted outcomes.
ROC Curve: The ROC curves for all models demonstrated strong discrimination ability, with areas under the curve indicating excellent performance.

Conclusion:
Overall, our NCAA March Madness prediction model performed exceptionally well, accurately predicting game outcomes based on team performance metrics.
