# Customer-Churn-Prediction

Customer Churn Prediction using Gradient Boosting

**Overview**

This project focuses on predicting customer churn for an online retail store using machine learning techniques. We implemented a Gradient Boosting Model (GBM) to analyze customer behavior and identify factors contributing to churn. The goal is to enable businesses to take proactive retention measures based on insights derived from the model.

*Steps in the Project*

1️⃣ Data Understanding <br>
The dataset consists of 33,859 customer records and 37 independent variables related to User Activity, Shopping Behavior, and Retail Category metrics. <br>
Class Imbalance: The dataset has 88.63% churn cases, indicating an imbalanced dataset. <br>
Data Distribution: Many features like clicks, daysInactiveAvg, and revenue are right-skewed, requiring preprocessing before modeling. <br>
2️⃣ Data Preprocessing & Feature Engineering <br>
Handling Outliers: Applied 1.5 × IQR method to cap extreme values and maintain data integrity. <br>
Feature Engineering: <br>
Created a new feature timeOfDayBin that categorizes purchase time into Night, Morning, Afternoon, and Evening. <br>
One-hot encoding was applied to categorical variables. <br>
Filtered out zero-variance features to reduce noise. <br>
Scaling: Standardized numerical features to ensure uniformity. <br>
Splitting Data: Used 70% for training and 30% for validation. <br>
3️⃣ Model Training <br>
Gradient Boosting Machines (GBM) <br>
Used caret::train() with 10-fold cross-validation. <br>
Compared two GBM models: <br>
Optimized for Accuracy <br>
Optimized for AUC (Area Under Curve) <br>
Final Model Selection: Chose the model with the highest AUC score (0.7571). <br>
Hyperparameter tuning using a grid search: <br>
gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5), <br>
                       n.trees = (1:3) * 50,<br>
                       shrinkage = c(0.01, 0.1),<br>
                       n.minobsinnode = c(10, 20))<br>
Variable Importance Analysis highlighted key factors impacting churn. <br>
4️⃣ Model Evaluation <br>
Validation Approach: <br>
10-fold cross-validation to prevent overfitting.
AUC (Area Under Curve) as the primary metric for performance.
Results: <br>
Final Model’s AUC: 0.7571 <br>
ROC Curve: Used for model comparison. <br>
5️⃣ Business Insights <br>
Key Features Contributing to Churn: <br>
Days Inactive: Customers inactive for long periods are more likely to churn. <br>
Visits & Clicks: Higher engagement correlates with lower churn. <br>
Variability in Usage: Irregular usage patterns indicate higher churn risk.<br>
Business Strategies to Reduce Churn:<br>
Implement reactivation campaigns targeting inactive users.<br>
Personalize the user experience based on visits and shopping behavior.<br>
Introduce incentives to boost customer engagement. <br>

**Results & Submission<br>**

Predictions were saved in submission_final.csv, containing customer IDs and their churn probabilities.
The model’s insights can help businesses design effective retention strategies. <br>

**Skills Used<br>**

Machine Learning: Supervised learning, predictive modeling, feature engineering.
Data Preprocessing: Outlier handling, missing value treatment, data transformation.
Feature Engineering: One-hot encoding, time-based segmentation.
Hyperparameter Tuning: Optimizing GBM with caret and grid search.
Model Evaluation: AUC, cross-validation, ROC curve analysis.
Visualization: Histograms, scatter plots, bar charts.
Business Analytics: Deriving actionable insights from customer data.
