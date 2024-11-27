# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
import os
print("Current working directory:", os.getcwd())

warnings.filterwarnings('ignore')  # Ignore warning messages to keep output clean

# Step 1: Data Collection (Using CSV File)
# Load your dataset from a CSV file
# The CSV file contains the features for the last 7 games
data = pd.read_csv('Celtics_Cavaliers.csv')  # Replace 'nba_data.csv' with your actual CSV file path

# Step 2: Label Creation
# Create labels for Over (1) or Under (0) based on Actual Score vs Betting Line
if 'Actual_Score' in data.columns and 'Betting_Line' in data.columns:
    data['Over_Under'] = (data['Actual_Score'] > data['Betting_Line']).astype(int)  # Label is 1 if actual score is greater than betting line
else:
    raise ValueError("'Actual_Score' or 'Betting_Line' is missing from the dataset.")  # Raise an error if required columns are missing

# Step 3: Feature Selection
# Select the features that will be used to train the model
features = [
    'Home_Team_PPG', 'Home_Team_Opp_PPG', 'Home_Team_Off_Eff', 'Home_Team_Def_Eff', 'Home_Team_Pace', 'Home_Team_eFG%', 'Home_Team_TS%', 
    'Home_Team_3PA', 'Home_Team_3P%', 'Home_Team_Turnover_Rate', 'Home_Team_Rebound_Rate', 'Home_Team_Rest_Days', 'Home_Team_Back_to_Back', 'Home_Team_Travel_Schedule',
    'Away_Team_PPG', 'Away_Team_Opp_PPG', 'Away_Team_Off_Eff', 'Away_Team_Def_Eff', 'Away_Team_Pace', 'Away_Team_eFG%', 'Away_Team_TS%', 
    'Away_Team_3PA', 'Away_Team_3P%', 'Away_Team_Turnover_Rate', 'Away_Team_Rebound_Rate', 'Away_Team_Rest_Days', 'Away_Team_Back_to_Back', 'Away_Team_Travel_Schedule',
    # New Feature
    'Home_Key_Player_Points', 'Home_Key_Player_Rebounds', 'Home_Key_Player_Assists', 'Home_Key_Player_Injured',
    'Away_Key_Player_Points', 'Away_Key_Player_Rebounds', 'Away_Key_Player_Assists', 'Away_Key_Player_Injured',
    # Interaction Terms
    'Home_Team_Pace_x_Home_Team_Off_Eff', 'Away_Team_Pace_x_Away_Team_Off_Eff'
]

# Ensure all selected features are present in the dataset
for feature in features:
    if feature not in data.columns:
        raise ValueError(f"Feature '{feature}' is missing from the dataset.")

# Step 4: Encoding Categorical Features
# Encode categorical features using Label Encoding
label_encoder = LabelEncoder()
categorical_features = ['Home_Team_Back_to_Back', 'Home_Team_Travel_Schedule', 'Away_Team_Back_to_Back', 'Away_Team_Travel_Schedule', 'Home_Key_Player_Injured', 'Away_Key_Player_Injured']

for feature in categorical_features:
    if data[feature].dtype == 'object':
        data[feature] = label_encoder.fit_transform(data[feature])  # Convert categorical values to numeric

# Step 5: Interaction Terms
# Create interaction terms to capture combined effects of features
# Home_Team_Pace * Home_Team_Off_Eff
data['Home_Team_Pace_x_Home_Team_Off_Eff'] = data['Home_Team_Pace'] * data['Home_Team_Off_Eff']
# Away_Team_Pace * Away_Team_Off_Eff
data['Away_Team_Pace_x_Away_Team_Off_Eff'] = data['Away_Team_Pace'] * data['Away_Team_Off_Eff']

# Step 6: Feature and Label Assignment
X = data[features]  # Features dataframe
y = data['Over_Under']  # Target labels

# Step 7: Data Splitting
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Hyperparameter Tuning with GridSearchCV for RandomForestClassifier
# Define hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200],  # Number of trees in the forest
    'max_depth': [10, 20, None],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10]  # Minimum number of samples required to split an internal node
}
rf = RandomForestClassifier(random_state=42)  # Initialize Random Forest model
# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
rf_best_model = grid_search.best_estimator_  # Best model from GridSearchCV

# Step 9: Predictions and Evaluation (Tuned Random Forest)
# Make predictions using the best Random Forest model
y_pred_rf_best = rf_best_model.predict(X_test)
# Evaluate the model's performance
rf_best_accuracy = accuracy_score(y_test, y_pred_rf_best)
rf_best_report = classification_report(y_test, y_pred_rf_best)

print("Tuned Random Forest Model Accuracy:", rf_best_accuracy)
print("Tuned Random Forest Classification Report:\n", rf_best_report)

# Step 10: Model Training with Gradient Boosting
# Train Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions using Gradient Boosting model
y_pred_gb = gb_model.predict(X_test)
# Evaluate the model's performance
gb_accuracy = accuracy_score(y_test, y_pred_gb)
gb_report = classification_report(y_test, y_pred_gb)

print("Gradient Boosting Model Accuracy:", gb_accuracy)
print("Gradient Boosting Classification Report:\n", gb_report)

# Step 11: Model Training with XGBoost
# Train XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Make predictions using XGBoost model
y_pred_xgb = xgb_model.predict(X_test)
# Evaluate the model's performance
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_report = classification_report(y_test, y_pred_xgb)

print("XGBoost Model Accuracy:", xgb_accuracy)
print("XGBoost Classification Report:\n", xgb_report)

# Step 12: Model Training with Neural Network (MLPClassifier)
# Train Neural Network (Multi-Layer Perceptron) model
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

# Make predictions using Neural Network model
y_pred_mlp = mlp_model.predict(X_test)
# Evaluate the model's performance
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
mlp_report = classification_report(y_test, y_pred_mlp)

print("Neural Network Model Accuracy:", mlp_accuracy)
print("Neural Network Classification Report:\n", mlp_report)

# Step 13: Feature Importance (XGBoost)
# Extract feature importance from XGBoost model
xgb_importance = xgb_model.feature_importances_
# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': xgb_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)  # Sort by importance

print("\nFeature Importance (XGBoost):\n", feature_importance_df)

# Note: The above implementation is a demonstration using CSV data.
# For a real-world model, accurate data collection from NBA sources and consistent updates are crucial.
