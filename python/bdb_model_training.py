# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:44:51 2023

@author: Dr. Nils Rosjat
"""
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from scipy.stats import uniform, randint
import xgboost as xgb

randomsearch = False # set to true to rerun RandomizedSearchCV

# fetch play data
play_data = pd.read_csv("data/plays.csv")

# fetch game data
game_data = pd.read_csv("data/games.csv")
game_data = game_data[['gameId','homeTeamAbbr','visitorTeamAbbr']]

# join play_data and game_data on gameId and playId
gamePlay_data = pd.merge(
    play_data,
    game_data,
    how="inner",
    left_on=["gameId"],
    right_on=["gameId"],
    sort=True,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    validate=None,
)

gamePlay_data['score_diff'] = np.where(
    gamePlay_data['possessionTeam'] == gamePlay_data['homeTeamAbbr'],
    gamePlay_data['preSnapHomeScore'] - gamePlay_data['preSnapVisitorScore'],
    np.where(
        gamePlay_data['possessionTeam'] == gamePlay_data['visitorTeamAbbr'],
        gamePlay_data['preSnapVisitorScore'] - gamePlay_data['preSnapHomeScore'],
        np.nan  # This handles cases where neither condition is true, you can replace np.nan with any default value you want
    )
)

# Convert gameclock to seconds
gamePlay_data['gameclock_seconds'] = gamePlay_data['gameClock'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))

# Calculate time_rem_qtr
gamePlay_data['time_rem_qtr'] = gamePlay_data['gameclock_seconds']

# Calculate time_rem_half
def time_remaining_half(row):
    if row['quarter'] == 1:
        return row['time_rem_qtr']  + 15*60
    elif row['quarter'] == 2:
        return row['time_rem_qtr']
    elif row['quarter'] == 3:
        return row['time_rem_qtr']  + 15*60
    else:
        return row['time_rem_qtr']

gamePlay_data['time_rem_half'] = gamePlay_data.apply(time_remaining_half, axis=1)

# Calculate time_rem_game
def time_remaining_game(row):
    if row['quarter'] == 1:
        return row['time_rem_qtr'] + 3*15*60
    elif row['quarter'] == 2:
        return row['time_rem_qtr'] + 2*15*60
    elif row['quarter'] == 3:
        return row['time_rem_qtr'] + 15*60
    else:
        return row['time_rem_qtr']

gamePlay_data['time_rem_game'] = gamePlay_data.apply(time_remaining_game, axis=1)

# Drop the temporary column 'gameclock_seconds'
gamePlay_data = gamePlay_data.drop('gameclock_seconds', axis=1)

columns_to_keep = ['gameId','playId','quarter','down','yardsToGo','playResult','absoluteYardlineNumber','defendersInTheBox','score_diff','time_rem_qtr','time_rem_half','time_rem_game']
gamePlay_data = gamePlay_data[columns_to_keep]

# Initialize an empty list to store individual dataframes
dfs = []

# Iterate over all files in the directory
for filename in os.listdir('data'):
    # Match only relevant filenames
    match = re.match(r"features_week_(\d+)_game_\d+.csv", filename)
    if match:
        print('Loading ' + filename)
        week = int(match.group(1))
        file_path = os.path.join('data', filename)
        
        # Load the file into a dataframe
        df = pd.read_csv(file_path)
        
        if 'LS1' in df.columns:
            print('Long Snapper Alarm!')
        
        # Add a column to store the week number
        df['week'] = week
        
        # Append the dataframe to the list
        dfs.append(df)
        print(filename + ' loaded')

# Combine all dataframes into a single dataframe
final_df = pd.concat(dfs, ignore_index=True)

# Filter rows based on gameId and playId
mask = (final_df['gameId'] == 2022091807) & (final_df['playId'] == 3597)

# Copy values from LS1 to Off11 and LS1_speed to Off_speed11
final_df.loc[mask, 'Off11'] = final_df.loc[mask, 'LS1']
final_df.loc[mask, 'Off_speed11'] = final_df.loc[mask, 'LS1_speed']

# Drop the columns LS1, LS1_speed, vx and vy
final_df = final_df.drop(columns=['LS1', 'LS1_speed'])

final_df = final_df.merge(gamePlay_data, on=['gameId', 'playId'], how='left')


# Split the data based on weeks
train_df = final_df[final_df['week'].isin([1, 2, 3, 4])]
#train_df = final_df[final_df['week'] == 1]
test_df = final_df[final_df['week'].isin([5,6])]

# Ensure the DataFrame is sorted correctly
test_df = test_df.sort_values(by=['gameId', 'playId', 'frameId'])

# Function to select the frames
def select_first_frames(group):
    return group.head(5)

# Function to select the frames
def select_last_frames(group):
    return group.tail(5)

# Group the DataFrame and apply the function
selected_first_df = test_df.groupby(['gameId', 'playId'], group_keys=False).apply(select_first_frames)
selected_last_df = test_df.groupby(['gameId', 'playId'], group_keys=False).apply(select_last_frames)
# Reset index to flatten the resulting DataFrame
selected_first_df = selected_first_df.reset_index(drop=True)
selected_last_df = selected_last_df.reset_index(drop=True)

predict_df = final_df[final_df['week'].isin([7, 8, 9])]

# Splitting the dataset into training and test sets
X_train = train_df.drop(['att_tackle','gameId','playId','nflId','displayName','frameId','week', 'defendersInTheBox', 'yardsToGo','quarter','down', 'playResult'], axis=1)
y_train = train_df['att_tackle']
X_test = test_df.drop(['att_tackle','gameId','playId','nflId','displayName','frameId','week', 'defendersInTheBox', 'yardsToGo','quarter','down', 'playResult'], axis=1)
y_test = test_df['att_tackle']
X_test_first = selected_first_df.drop(['att_tackle','gameId','playId','nflId','displayName','frameId','week', 'defendersInTheBox', 'yardsToGo','quarter','down', 'playResult'], axis=1)
y_test_first = selected_first_df['att_tackle']
X_test_last = selected_last_df.drop(['att_tackle','gameId','playId','nflId','displayName','frameId','week', 'defendersInTheBox', 'yardsToGo','quarter','down', 'playResult'], axis=1)
y_test_last = selected_last_df['att_tackle']
X_pred = predict_df.drop(['att_tackle','gameId','playId','nflId','displayName','frameId','week', 'defendersInTheBox', 'yardsToGo','quarter','down', 'playResult'], axis=1)
y_pred = predict_df['att_tackle']

###################################### Randomized search for xgboost model ################
if randomsearch:
    # Define the parameter grid
    param_dist = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'gamma': uniform(0, 1),
        'min_child_weight': randint(1, 10),
        'scale_pos_weight': [sum(y_train == 0) / sum(y_train == 1)]
    }
    
    # Initialize the XGBClassifier
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', booster='gbtree', random_state=42, alpha=0.1, reg_lambda=1.0)
    
    # Initialize RandomizedSearchCV
    rs_clf = RandomizedSearchCV(xgb_clf, param_distributions=param_dist, n_iter=10, scoring='roc_auc', cv=5, random_state=42, verbose=2, n_jobs = 1)
    
    # Fit RandomizedSearchCV
    rs_clf.fit(X_train, y_train)
    
    # Print the best parameters and the corresponding AUC score
    best_params = rs_clf.best_params_
    best_score = rs_clf.best_score_
    print(f"Best parameters: {best_params}")
    print(f"Best AUC score: {best_score:.4f}")
    
    # Train the best model on the full training data
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)
    
    # Predict on the test data
    y_test_pred = best_model.predict(X_test)
    
    from sklearn.metrics import precision_score, f1_score, roc_auc_score
    
    # Calculate the metrics for the best model
    precision_test = precision_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)
    roc_auc_test = roc_auc_score(y_test, y_test_pred)
    
    # Print the metrics for the best model
    print(f"Precision on test data: {precision_test:.4f}")
    print(f"F1-Score on test data: {f1_test:.4f}")
    print(f"ROC-AUC on test data: {roc_auc_test:.4f}")
    
    ###############################################################################
    ############# Randomized Search round 2 #######################################
    
    from scipy.stats import randint, uniform
    
    # Define the parameter grid
    param_dist = {
        'n_estimators': randint(350, 370),
        'max_depth': randint(4, 7),
        'learning_rate': uniform(0.1, 0.15),
        'subsample': uniform(0.9, 1.0),
        'colsample_bytree': uniform(0.5, 0.7),
        'gamma': uniform(0.45, 0.46),
        'min_child_weight': randint(5, 7),
        'scale_pos_weight': uniform(7.5, 7.6),
    }
    
    # Initialize the XGBClassifier
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', booster='gbtree', random_state=42, alpha=0.1, reg_lambda=1.0)
    
    # Initialize RandomizedSearchCV
    rs_clf = RandomizedSearchCV(xgb_clf, param_distributions=param_dist, n_iter=10, scoring='roc_auc', cv=5, random_state=42, verbose=1, n_jobs=1)
    
    # Fit RandomizedSearchCV
    rs_clf.fit(X_train, y_train)
    
    # Print the best parameters
    print("Best parameters:", rs_clf.best_params_)
    
################# XGBoost #####################################################

# Convert training data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)

dval = xgb.DMatrix(X_test, label=y_test)
evals = [(dtrain, 'train'), (dval, 'eval')]

# XGBoost parameters after hyperpar. tuning
params_auc = {
    'alpha': 0.1,  # L1 regularization term on weights
    'booster': 'gbtree',  # Use tree-based models
    'colsample_bytree': 0.7621780831931537,  # Subsample ratio of columns when constructing each tree
    'eval_metric': 'auc',  # Evaluation metric for validation data
    'gamma': 0.8873285809485615,  # Minimum loss reduction required to make a further partition
    'lambda': 1.0,  # L2 regularization term on weights
    'learning_rate': 0.20979909127171076,  # Step size shrinkage used in update to prevents overfitting
    'max_depth': 4,  # Maximum depth of a tree
    'min_child_weight': 5,  # Minimum sum of instance weight (hessian) needed in a child
    'n_estimators': 356,  # Number of gradient boosted trees
    'objective': 'binary:logistic',  # Specify the learning task and the corresponding learning objective
    'random_state': 42,  # Random number seed
    'scale_pos_weight': 10.888328921687293,  # Balancing of positive and negative weights
    'subsample': 0.9999749158180029  # Subsample ratio of the training instance
}

# Train the XGBoost model
num_rounds = 100
bst = xgb.train(params_auc, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=10, verbose_eval=True)

# Predict on training data to get the recall score
y_train_pred = bst.predict(dtrain)
y_train_pred = [1 if p > 0.5 else 0 for p in y_train_pred]  # Convert probabilities to class labels

# Get the recall score
recall = recall_score(y_train, y_train_pred)
print(f"Recall on training data: {recall:.4f}")

# Convert training data to DMatrix format for XGBoost
dtest = xgb.DMatrix(X_test, label=y_test)

y_test_pred = bst.predict(dtest)
y_test_pred = [1 if p > 0.5 else 0 for p in y_test_pred]
y_test_pred2 = [1 if p > 0.6 else 0 for p in y_test_pred]

# Convert training data to DMatrix format for XGBoost
dtest_first = xgb.DMatrix(X_test_first, label=y_test_first)

y_test_pred_first = bst.predict(dtest_first)
y_test_pred_first = [1 if p > 0.5 else 0 for p in y_test_pred_first]

# Convert training data to DMatrix format for XGBoost
dtest_last = xgb.DMatrix(X_test_last, label=y_test_last)

y_test_pred_last = bst.predict(dtest_last)
y_test_pred_last = [1 if p > 0.5 else 0 for p in y_test_pred_last]

# Get the recall score
recall = recall_score(y_test, y_test_pred)
print(f"Recall on test data: {recall:.4f}")

# Plot the feature importance graphic
xgb.plot_importance(bst)
plt.show()

# Run 5-fold cross validation
cv_scores = cross_val_score(xgb.XGBClassifier(**params_auc), X_train, y_train, cv=5, scoring='recall')
print(f"Cross-validation recall scores: {cv_scores}")
print(f"Mean CV recall: {cv_scores.mean():.4f}")

precision_train = precision_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)
roc_auc_train = roc_auc_score(y_train, y_train_pred)

print(f"Precision on training data: {precision_train:.4f}")
print(f"F1-Score on training data: {f1_train:.4f}")
print(f"ROC-AUC on training data: {roc_auc_train:.4f}")

precision_test = precision_score(y_test, y_test_pred2)
f1_test = f1_score(y_test, y_test_pred2)
roc_auc_test = roc_auc_score(y_test, y_test_pred2)

print(f"Precision on test data: {precision_test:.4f}")
print(f"F1-Score on test data: {f1_test:.4f}")
print(f"ROC-AUC on test data: {roc_auc_test:.4f}")

precision_test_first = precision_score(y_test_first, y_test_pred_first)
f1_test_first = f1_score(y_test_first, y_test_pred_first)
roc_auc_test_first = roc_auc_score(y_test_first, y_test_pred_first)
recall_first = recall_score(y_test_first, y_test_pred_first)

print(f"Recall on test data (first 5 frames): {recall_first:.4f}")
print(f"Precision on test data (first 5 frames): {precision_test_first:.4f}")
print(f"F1-Score on test data (first 5 frames): {f1_test_first:.4f}")
print(f"ROC-AUC on test data (first 5 frames): {roc_auc_test_first:.4f}")

precision_test_last = precision_score(y_test_last, y_test_pred_last)
f1_test_last = f1_score(y_test_last, y_test_pred_last)
roc_auc_test_last = roc_auc_score(y_test_last, y_test_pred_last)
recall_last = recall_score(y_test_last, y_test_pred_last)

print(f"Recall on training data (last 5 frames): {recall_last:.4f}")
print(f"Precision on test data (last 5 frames): {precision_test_last:.4f}")
print(f"F1-Score on test data (last 5 frames): {f1_test_last:.4f}")
print(f"ROC-AUC on test data (last 5 frames): {roc_auc_test_last:.4f}")

# Convert training data to DMatrix format for XGBoost
dpred = xgb.DMatrix(X_pred)
y_probs = bst.predict(dpred)

predict_df['tackle_prob'] = y_probs
predict_df['tackle_binary'] = (predict_df['tackle_prob'] > 0.75).astype(int)

predict_df.to_excel('predictions.xlsx',index=False)