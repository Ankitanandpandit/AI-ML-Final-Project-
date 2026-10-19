# AI-ML-Final-Project-
AI &amp; ML Final Project 

What You'll Do
• Ek complete AI / Machine Learning project banao jo aapne seekhe skills use kare
• GitHub pe publish karo with proper README file
• Apna work review karke portal pe final project submit karo
Final Project Ideas
Project Description
1. Student Performance Predictor Previous marks se final score predict (Regression)
2. Sentiment Analysis Movie reviews positive/negative classify
3. House Price Prediction Features se price predict (Regression)
4. Spam Email Classifier Email text se spam/ham detect
5. Expense Category Predictor Description se category predict
Recommended: Student Score Prediction
# Final Project: Student Score Prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
data = {
 'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9, 1, 5.5, 3.5, 6.5, 7.5, 4.5, 8.5],
 'Previous_Score': [50, 55, 60, 65, 70, 75, 80, 85, 40, 68, 58, 72, 78, 63, 82],
 'Final_Score': [55, 60, 65, 72, 78, 82, 88, 92, 45, 75, 62, 80, 85, 70, 90]
}
df = pd.DataFrame(data)
X = df[['Hours_Studied', 'Previous_Score']]
y = df['Final_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
# New student prediction
new = [[6, 70]] # 6 hours, previous 70
print(f"Predicted Score: {model.predict(new)[0]:.1f}")
GitHub README Template
# Student Score Prediction - Final Project
## Overview
Predicts final exam score based on hours studied and previous scores using Linear Regression.