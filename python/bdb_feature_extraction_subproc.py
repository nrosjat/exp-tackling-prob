# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:07:40 2023

@author: Lagor
"""

import numpy as np
import pandas as pd
import os
import re

# what about projected distance to ballcarrier in next frame?

def calculate_deviation(x1, y1, x2, y2, player_direction_deg):
    # Adjust the angle based on your definition
    direct_angle_rad = np.arctan2(y1 - y2, x2 - x1)  # Note the inversion in y-values and the order of x-values
    
    # Convert the angle to degrees and adjust for your specific definition
    direct_angle_deg = np.rad2deg(direct_angle_rad)
    direct_angle_deg = (direct_angle_deg + 360) % 360  # Ensure the angle is between 0 and 360
    
    # Calculate the deviation
    deviation_deg = player_direction_deg - direct_angle_deg
    deviation_deg = (deviation_deg + 180) % 360 - 180  # Ensure the deviation is between -180 and 180
    
    return deviation_deg

#def compute_speeds(week):
#    # Compute speed components
#    week['vx'] = week['s'] * np.cos(np.deg2rad(week['dir']))
#    week['vy'] = week['s'] * np.sin(np.deg2rad(week['dir']))
#
#    # Create unique positions without innermost loop
#    week['pos_unique'] = (week['position']
#                          .add(week
#                               .groupby(['gameId', 'playId', 'position'])
#                               .cumcount()
#                               .add(1)))
#
#    # Compute pairwise differences in speed components using broadcasting
#    vx = week['vx'].values[:, np.newaxis] - week['vx'].values
#    vy = week['vy'].values[:, np.newaxis] - week['vy'].values
#
#    # Compute the magnitude of these relative speeds
#    relative_speeds = np.sqrt(vx**2 + vy**2)
#
#    # Create a DataFrame for the relative speeds
#    speeds_df = pd.DataFrame(relative_speeds, index=week['nflId'], columns=week['pos_unique'].fillna('football'))
#
#    # Reset index to pop out nflId into its own column
#    speeds_df = speeds_df.reset_index()
#
#    # Merge new speed values onto the original dataframe
#    week = week.merge(speeds_df, on='nflId', suffixes=('', '_speed'))
#
#    return week

def compute_speeds(week): # new to be corrected matched to pipeline?
    # Define a function to compute relative speeds for each group
    def compute_relative_speeds_for_group(frame):
        # Compute pairwise speed differences directly
        speeds = frame['s'].values
        relative_speeds_directional = speeds[:, None] - speeds
        
        # Create a DataFrame for the relative speeds
        speeds_df = pd.DataFrame(relative_speeds_directional, 
                                 index=frame['nflId'], 
                                 columns=frame['pos_unique'].fillna('football'))

        # Reset index to pop out nflId into its own column
        speeds_df = speeds_df.reset_index()

        # Merge new speed values onto the frame
        frame = frame.merge(speeds_df, on='nflId', suffixes=('', '_speed'))
        
        return frame

    # Apply the function to each group and concatenate the results
    week = week.groupby(['frameId', 'playId']).apply(compute_relative_speeds_for_group).reset_index(drop=True)

    return week

# Function to get the value from the column corresponding to ballCarrier's pos_unique value
def get_dist(row, tracking_df):
    gameId, playId = row['gameId'], row['playId']
    ballCarrierId = ballCarrier_map.get((gameId, playId), None)
    if ballCarrierId is None:
        return None
    ballCarrier_pos = tracking_df.loc[(tracking_df['gameId'] == gameId) & 
                                       (tracking_df['playId'] == playId) & 
                                       (tracking_df['nflId'] == ballCarrierId), 'pos_unique'].iloc[0]
    return row[ballCarrier_pos]

# Adjusted Step 3: Count offensive and defensive players closer to the ball within the same game, play, and frame
def count_players(row,tracking_df):
    # Filter for the current frame, game, and play
    current_data = tracking_df[(tracking_df['frameId'] == row['frameId']) & 
                               (tracking_df['gameId'] == row['gameId']) & 
                               (tracking_df['playId'] == row['playId'])]
    
    # Count offensive players closer to the ball
    offensive_count = sum((current_data['club'] == row['possessionTeam']) & (current_data['dist_ballCarrier'] < row['dist_ballCarrier']))
    
    # Count defensive players closer to the ball
    defensive_count = sum((current_data['club'] != row['possessionTeam']) & (current_data['club'] != 'football') & (current_data['dist_ballCarrier'] < row['dist_ballCarrier']))
    
    return pd.Series([offensive_count, defensive_count], index=['offensive_players_closer', 'defensive_players_closer'])


def compute_mean_distances(row):
    # If the row represents 'football', return 0 for both mean distances
    if row['club'] == 'football':
        return pd.Series([0, 0], index=['mean_dist_to_offense', 'mean_dist_to_defense'])

    offense_positions = ['QB', 'WR', 'TE', 'RB', 'FB', 'T', 'G', 'C']
    defense_positions = ['DT', 'DE', 'CB', 'SS', 'FS', 'ILB', 'OLB', 'MLB', 'NT', 'DB']
    
    is_offensive = row['club'] == row['possessionTeam']
    
    # Get the list of columns corresponding to offensive and defensive positions excluding NaNs, 'club', and 'possessionTeam'
    offensive_columns = [col for col in row.index if any(pos in col for pos in offense_positions) and col not in ['dist_ballCarrier','pff_missedTackle','club', 'possessionTeam'] and not pd.isnull(row[col])]
    defensive_columns = [col for col in row.index if any(pos in col for pos in defense_positions) and col not in ['dist_ballCarrier','pff_missedTackle','club', 'possessionTeam'] and not pd.isnull(row[col])]

    # Filter and sort the offensive and defensive distances, ensuring they are floats
    offensive_distances = row[offensive_columns].dropna().astype(float).sort_values()
    defensive_distances = row[defensive_columns].dropna().astype(float).sort_values()

    # Exclude the closest player if the current player is offensive/defensive
    if is_offensive:
        offensive_distances = offensive_distances.iloc[1:5]
    else:
        defensive_distances = defensive_distances.iloc[1:5]

    # Calculate the mean distances for offense and defense
    mean_offensive_distance = offensive_distances.head(4).mean()
    mean_defensive_distance = defensive_distances.head(4).mean()

    return pd.Series([mean_offensive_distance, mean_defensive_distance], index=['mean_dist_to_offense', 'mean_dist_to_defense'])

#for file in tracking_datafile:
#    
#    tracking_data = pd.read_csv(file)
#    fname = file[:-4] + '_ori.csv'
#    # Apply transformations based on playDirection
#    mask = tracking_data['playDirection'] == 'left'
#    
#    tracking_data.loc[mask, 'x'] = 120 - tracking_data.loc[mask, 'x']
#    tracking_data.loc[mask, 'y'] = (160 / 3) - tracking_data.loc[mask, 'y']
#    tracking_data.loc[mask, 'o'] = 180 + tracking_data.loc[mask, 'o']
#    tracking_data.loc[mask, 'o'] = tracking_data.loc[mask, 'o'].apply(lambda x: x - 360 if x > 360 else x)
#    tracking_data.loc[mask, 'dir'] = 180 + tracking_data.loc[mask, 'dir']
#    tracking_data.loc[mask, 'dir'] = tracking_data.loc[mask, 'dir'].apply(lambda x: x - 360 if x > 360 else x)
#    
#    tracking_data.to_csv(fname)
    
# fetch play data
play_data = pd.read_csv("data/plays.csv")

# fetch tackle data
tackle_data = pd.read_csv("data/tackles.csv")
tackle_data['att_tackle'] = tackle_data['tackle'] + tackle_data['assist']# + tackle_data['pff_missedTackle']

# join play_data and game_data on gameId and playId
tacklePlay_data = pd.merge(
    play_data,
    tackle_data,
    how="outer",
    left_on=["gameId","playId"],
    right_on=["gameId","playId"],
    sort=True,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    validate=None,
)

tacklePlay_data = tacklePlay_data[['gameId','playId','ballCarrierId','possessionTeam','nflId','tackle','assist','forcedFumble','pff_missedTackle','att_tackle']]


tracking_datafile = [
                    "data/tracking_dist_week_1_ori.csv",
                    "data/tracking_dist_week_2_ori.csv",
                    "data/tracking_dist_week_3_ori.csv",
                    "data/tracking_dist_week_4_ori.csv",
                    "data/tracking_dist_week_5_ori.csv",
                    "data/tracking_dist_week_6_ori.csv",
                    "data/tracking_dist_week_7_ori.csv",
                    "data/tracking_dist_week_8_ori.csv",
                    "data/tracking_dist_week_9_ori.csv"
                    ]

for file in tracking_datafile:
    
    print('Working on ' + file)    
    tracking_data = pd.read_csv(file)
    
    # Group the tracking data by gameId
    grouped = tracking_data.groupby('gameId')

    for game_id, game_data in grouped:
        print(f'Processing game: {game_id}')
        
        # Merge the DataFrames on gameId, playId, and nflId
        tracking_df = pd.merge(game_data, 
                           tacklePlay_data[['gameId', 'playId', 'nflId', 'tackle', 'assist', 'forcedFumble', 'pff_missedTackle', 'att_tackle', 'possessionTeam']], 
                           on=['gameId', 'playId', 'nflId'], 
                           how='left')
        
        fname = f"{file[:-4]}_game_{game_id}_feature.csv"
        
        # Check if file exists
        if os.path.exists(fname):
            print("File exists, skip")
            continue
        # Step 1: Replace NaN in possessionTeam
        try:
            most_freq_possession = tracking_df.groupby('playId')['possessionTeam'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'noTeam')
            # Fill NaN values in possessionTeam using the computed most frequent values
            tracking_df['possessionTeam'] = tracking_df.groupby('playId')['possessionTeam'].transform(lambda x: x.fillna(most_freq_possession.get(x.name, 'noTeam')))
        except KeyError:
            print(f'Error with: {game_id}')
            continue
        
        
        print('Finished possession team.')    
        
        tracking_df = tracking_df[tracking_df['possessionTeam'] != 'noTeam']
        # Step 2: Add a new column dist_ballCarrier
        # Mapping of (gameId, playId) to ballCarrierId
        ballCarrier_map = tacklePlay_data.set_index(['gameId', 'playId'])['ballCarrierId'].to_dict()
        
        
        # Add the dist_ballCarrier column to the merged DataFrame
        tracking_df['dist_ballCarrier'] = tracking_df.apply(lambda row: get_dist(row, tracking_df), axis=1)
        
        print('Finished ball carrier distance.')    
        
        
        # Step 2: Replace NaN in tackle-related columns
        cols_to_replace_nan = ['tackle', 'assist', 'forcedFumble', 'pff_missedTackle', 'att_tackle']
        tracking_df[cols_to_replace_nan] = tracking_df[cols_to_replace_nan].fillna(0)
        print('Finished replace nan for tackles.')    
        
        
        tracking_df[['offensive_players_closer', 'defensive_players_closer']] = tracking_df.apply(lambda row: count_players(row, tracking_df), axis=1)
        print('Finished closer players.')    
    
        
        tracking_df[['mean_dist_to_offense', 'mean_dist_to_defense']] = tracking_df.apply(compute_mean_distances, axis=1)
        print('Finished mean distance closest 4.')    
        
        # Optimized speed calculations
        tracking_df = compute_speeds(tracking_df)
        print('Finished computing speeds')
        
        tracking_df['is_ballcarrier'] = tracking_df.apply(lambda row: row['nflId'] == ballCarrier_map.get((row['gameId'], row['playId']), None), axis=1)
        ballcarrier_df = tracking_df[tracking_df['is_ballcarrier']][['gameId', 'playId', 'frameId', 'x', 'y']].rename(columns={'x': 'ballcarrier_x', 'y': 'ballcarrier_y'})
        merged_df = pd.merge(tracking_df, ballcarrier_df, on=['gameId', 'playId', 'frameId'], how='left')
        merged_df['deviation'] = merged_df.apply(lambda row: calculate_deviation(row['x'], row['y'], row['ballcarrier_x'], row['ballcarrier_y'], row['dir']), axis=1)
    
        merged_df.to_csv(fname)
        print('Finished file ' + file)

offense_positions = ['QB', 'WR', 'TE', 'RB', 'FB', 'T', 'G', 'C', 'LS']
defense_positions = ['DT', 'DE', 'CB', 'SS', 'FS', 'ILB', 'OLB', 'MLB', 'NT', 'DB']

def get_unique_name(base_name, existing_cols):
    count = 1
    new_name = base_name
    while new_name in existing_cols:
        count += 1
        new_name = base_name[:-3] + str(int(base_name[-3:]) + count)
    return new_name

def rename_file(filename):
    # Match the given filename pattern
    pattern = r"tracking_dist_(week_\d+)_ori_game_(\d+)_feature.csv"
    match = re.match(pattern, filename)

    if match:
        # Extract the matched groups
        week = match.group(1)
        game_date = match.group(2)
        
        # Construct the new filename
        new_filename = f"features_{week}_game_{game_date}.csv"
        return new_filename
    else:
        return filename
    
def correct_mislabeled_positions(group):
    for _, row in group.iterrows():
        # Mislabeled offensive player for the non-possession team
        if row['club'] != row['possessionTeam'] and row['position'] in offense_positions:
            mislabel_col = row['pos_unique']
            mislabel_col2 = row['pos_unique'] + '_speed'
            #if mislabel_col in group.columns:
            #    group.rename(columns={mislabel_col: 'DB100'}, inplace=True)
            #if mislabel_col2 in group.columns:
            #    group.rename(columns={mislabel_col2: 'DB100_speed'}, inplace=True)
            if mislabel_col in group.columns:
                unique_name = get_unique_name('DB100', group.columns)
                group.rename(columns={mislabel_col: unique_name}, inplace=True)
                unique_name_speed = unique_name + '_speed'
                group.rename(columns={mislabel_col2: unique_name_speed}, inplace=True)

        # Mislabeled defensive player for the possession team
        elif row['club'] == row['possessionTeam'] and row['position'] in defense_positions:
            mislabel_col = row['pos_unique']
            mislabel_col2 = row['pos_unique'] + '_speed'
            #if mislabel_col in group.columns:
            #    group.rename(columns={mislabel_col: 'WR100'}, inplace=True)
            #if mislabel_col2 in group.columns:
            #    group.rename(columns={mislabel_col2: 'WR100_speed'}, inplace=True)
            if mislabel_col in group.columns:
                unique_name = get_unique_name('WR100', group.columns)
                group.rename(columns={mislabel_col: unique_name}, inplace=True)
                unique_name_speed = unique_name + '_speed'
                group.rename(columns={mislabel_col2: unique_name_speed}, inplace=True)
    return group

# Get list of all files in 'data' subfolder ending with 'features.csv'
file_list = [f for f in os.listdir('data') if f.endswith('feature.csv')]

#dfs = []
for file in file_list:
    fname_new = os.path.join('data', rename_file(file))
    
    if not os.path.isfile(fname_new):
        df = pd.read_csv(os.path.join('data', file))
        
        # Remove first 4 and last 5 frames for each play in each game
        to_remove = df.groupby(['gameId', 'playId']).apply(lambda x: x.head(4).index.append(x.tail(5).index))
        df = df.drop(index=to_remove.explode().values)
        
        # The provided code goes here...
        # Group by gameId and playId and apply the correction function
        df = df.groupby(['gameId', 'playId']).apply(correct_mislabeled_positions).reset_index(drop=True)
    
    
    
        # Create Off, Def, Off_speed and Def_speed columns initialized with NaN
        for i in range(1, 12):
            df[f'Off{i}'] = np.nan
            df[f'Def{i}'] = np.nan
            df[f'Off_speed{i}'] = np.nan
            df[f'Def_speed{i}'] = np.nan
    
        # Loop over each row
        for index, row in df.iterrows():
            i, j, k, l = 1, 1, 1, 1
            for col in df.columns:
                if any(re.match(f"^{pos}\d+$", col) for pos in offense_positions) and not pd.isna(row[col]):
                    df.at[index, f'Off{i}'] = row[col]
                    i += 1
                elif any(re.match(f"^{pos}\d+$", col) for pos in defense_positions) and not pd.isna(row[col]):
                    df.at[index, f'Def{j}'] = row[col]
                    j += 1
                elif any(re.match(f"^{pos}\d+_speed$", col) for pos in offense_positions) and not pd.isna(row[col]):
                    df.at[index, f'Off_speed{k}'] = row[col]
                    k += 1
                elif any(re.match(f"^{pos}\d+_speed$", col) for pos in defense_positions) and not pd.isna(row[col]):
                    df.at[index, f'Def_speed{l}'] = row[col]
                    l += 1
    
        # Drop the original columns
        cols_to_drop = [col for col in df.columns if any(re.match(f"^{pos}\d+$", col) or re.match(f"^{pos}\d+_speed$", col) for pos in offense_positions + defense_positions)]
        cols_to_drop += ['time','playDirection','a','dis','o','dir','pos_unique', 'is_ballcarrier','ballcarrier_x','ballcarrier_y']
        df = df.drop(columns=cols_to_drop)
    
        # Reorder columns
        cols_order = [f'Off{i}' for i in range(1, 12)] + [f'Def{i}' for i in range(1, 12)] + [f'Off_speed{i}' for i in range(1, 12)] + [f'Def_speed{i}' for i in range(1, 12)]
        remaining_cols = [col for col in df.columns if col not in cols_order]
        df = df[remaining_cols + cols_order]
        df = df[(df.club != df.possessionTeam) & (df.frameId > 4) & (df.displayName != 'football')]
        cols_to_drop2 = ['Unnamed: 0.1','Unnamed: 0','jerseyNumber','club','event','position','tackle', 'assist','forcedFumble','pff_missedTackle','possessionTeam']
        df = df.drop(columns=cols_to_drop2)
        
        df.to_csv(fname_new,index=False)
    else:
        print('File already exists!')

#    dfs.append(df)

# Merge all processed dataframes
#final_df = pd.concat(dfs, axis=0, ignore_index=True)

# Save to csv
#final_df.to_csv('tracking_features.csv', index=False)


"""
# Group by gameId and playId and apply the correction function
df = df.groupby(['gameId', 'playId']).apply(correct_mislabeled_positions).reset_index(drop=True)



# Create Off, Def, Off_speed and Def_speed columns initialized with NaN
for i in range(1, 12):
    df[f'Off{i}'] = np.nan
    df[f'Def{i}'] = np.nan
    df[f'Off_speed{i}'] = np.nan
    df[f'Def_speed{i}'] = np.nan

# Loop over each row
for index, row in df.iterrows():
    i, j, k, l = 1, 1, 1, 1
    for col in df.columns:
        if any(re.match(f"^{pos}\d+$", col) for pos in offense_positions) and not pd.isna(row[col]):
            df.at[index, f'Off{i}'] = row[col]
            i += 1
        elif any(re.match(f"^{pos}\d+$", col) for pos in defense_positions) and not pd.isna(row[col]):
            df.at[index, f'Def{j}'] = row[col]
            j += 1
        elif any(re.match(f"^{pos}\d+_speed$", col) for pos in offense_positions) and not pd.isna(row[col]):
            df.at[index, f'Off_speed{k}'] = row[col]
            k += 1
        elif any(re.match(f"^{pos}\d+_speed$", col) for pos in defense_positions) and not pd.isna(row[col]):
            df.at[index, f'Def_speed{l}'] = row[col]
            l += 1

# Drop the original columns
cols_to_drop = [col for col in df.columns if any(re.match(f"^{pos}\d+$", col) or re.match(f"^{pos}\d+_speed$", col) for pos in offense_positions + defense_positions)]
cols_to_drop += ['time','playDirection','a','dis','o','dir','pos_unique', 'is_ballcarrier','ballcarrier_x','ballcarrier_y']
df = df.drop(columns=cols_to_drop)

# Reorder columns
cols_order = [f'Off{i}' for i in range(1, 12)] + [f'Def{i}' for i in range(1, 12)] + [f'Off_speed{i}' for i in range(1, 12)] + [f'Def_speed{i}' for i in range(1, 12)]
remaining_cols = [col for col in df.columns if col not in cols_order]
df = df[remaining_cols + cols_order]
df = df[(df.club != df.possessionTeam) & (df.frameId > 4)]

test = df[df.att_tackle == 1].copy()
print(df)
"""

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

########### Random Forest #####################################################

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, recall_score
from scipy.stats import uniform, randint

# Splitting the dataset into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train = train_df.drop(['att_tackle','gameId','playId','nflId','displayName','frameId','week', 'defendersInTheBox', 'yardsToGo','quarter','down', 'offenseFormation', 'playResult', 'ballCarrierId'], axis=1)
y_train = train_df['att_tackle']
X_test = test_df.drop(['att_tackle','gameId','playId','nflId','displayName','frameId','week', 'defendersInTheBox', 'yardsToGo','quarter','down', 'offenseFormation', 'playResult', 'ballCarrierId'], axis=1)
y_test = test_df['att_tackle']
X_test_first = selected_first_df.drop(['att_tackle','gameId','playId','nflId','displayName','frameId','week', 'defendersInTheBox', 'yardsToGo','quarter','down', 'offenseFormation', 'playResult', 'ballCarrierId'], axis=1)
y_test_first = selected_first_df['att_tackle']
X_test_last = selected_last_df.drop(['att_tackle','gameId','playId','nflId','displayName','frameId','week', 'defendersInTheBox', 'yardsToGo','quarter','down', 'offenseFormation', 'playResult', 'ballCarrierId'], axis=1)
y_test_last = selected_last_df['att_tackle']
X_pred = predict_df.drop(['att_tackle','gameId','playId','nflId','displayName','frameId','week', 'defendersInTheBox', 'yardsToGo','quarter','down', 'offenseFormation', 'playResult', 'ballCarrierId'], axis=1)
y_pred = predict_df['att_tackle']

X_train2 = train_df[['defensive_players_closer','offensive_players_closer','football_speed','football','dist_ballCarrier','x','y','s','mean_dist_to_offense','mean_dist_to_defense','deviation','absoluteYardlineNumber','time_rem_game','time_rem_qtr','score_diff','time_rem_half']]
X_test2 = test_df[['defensive_players_closer','offensive_players_closer','football_speed','football','dist_ballCarrier','x','y','s','mean_dist_to_offense','mean_dist_to_defense','deviation','absoluteYardlineNumber','time_rem_game','time_rem_qtr','score_diff','time_rem_half']]
X_pred2 = predict_df[['defensive_players_closer','offensive_players_closer','football_speed','football','dist_ballCarrier','x','y','s','mean_dist_to_offense','mean_dist_to_defense','deviation','absoluteYardlineNumber','time_rem_game','time_rem_qtr','score_diff','time_rem_half']]

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize a random forest classifier
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

#grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
#                           cv=5, n_jobs=1, verbose=2, scoring='recall')

#grid_search.fit(X_train, y_train)
#best_params = grid_search.best_params_
#best_model = grid_search.best_estimator_

# Initialize a random forest classifier
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Train the model
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Get the recall score
recall = recall_score(y_test, y_pred)
print(f"Recall on training data: {recall:.4f}")

"""
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Classification Report (includes precision, recall, and F1-score)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC AUC Score
y_prob = rf.predict_proba(X_test)[:, 1]  # probabilities for the positive class
roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC AUC Score: {roc_auc:.2f}")

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
# Get feature importances
importances = rf.feature_importances_

# Get the feature names
feature_names = X_train.columns

# Sort the features by importance
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
"""

################# XGBoost #####################################################

import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Convert training data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)

dval = xgb.DMatrix(X_test, label=y_test)  # Assuming you've split a validation set
evals = [(dtrain, 'train'), (dval, 'eval')]

# Define XGBoost parameters
params_auc_orig = {
    'objective': 'binary:logistic',  # Binary classification problem
    'eval_metric': 'auc',        # Logarithmic loss
    'booster': 'gbtree',             # Tree-based models
    'random_state': 42,
    'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1),  # Handle class imbalance
    'alpha': 0.1, # L1 prevent overfitting 
    'lambda': 1.0 # L2 prevent overfitting
}

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


# Define XGBoost parameters
params_lloss = {
    'objective': 'binary:logistic',  # Binary classification problem
    'eval_metric': 'logloss',        # Logarithmic loss
    'booster': 'gbtree',             # Tree-based models
    'random_state': 42,
    'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1),  # Handle class imbalance
    'alpha': 0.1, # L1 prevent overfitting 
    'lambda': 1.0 # L2 prevent overfitting
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

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Get the recall score
recall = recall_score(y_test, y_test_pred)
print(f"Recall on training data: {recall:.4f}")

xgb.plot_importance(bst)
plt.show()

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(xgb.XGBClassifier(**params_auc), X_train, y_train, cv=5, scoring='recall')
print(f"Cross-validation recall scores: {cv_scores}")
print(f"Mean CV recall: {cv_scores.mean():.4f}")

from sklearn.metrics import precision_score, f1_score, roc_auc_score

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

print(f"Recall on training data: {recall_first:.4f}")
print(f"Precision on test data: {precision_test_first:.4f}")
print(f"F1-Score on test data: {f1_test_first:.4f}")
print(f"ROC-AUC on test data: {roc_auc_test_first:.4f}")

precision_test_last = precision_score(y_test_last, y_test_pred_last)
f1_test_last = f1_score(y_test_last, y_test_pred_last)
roc_auc_test_last = roc_auc_score(y_test_last, y_test_pred_last)
recall_last = recall_score(y_test_last, y_test_pred_last)

print(f"Recall on training data: {recall_last:.4f}")
print(f"Precision on test data: {precision_test_last:.4f}")
print(f"F1-Score on test data: {f1_test_last:.4f}")
print(f"ROC-AUC on test data: {roc_auc_test_last:.4f}")

# Convert training data to DMatrix format for XGBoost
dpred = xgb.DMatrix(X_pred)
y_probs = bst.predict(dpred)

predict_df['tackle_prob'] = y_probs #[:,1]
predict_df['tackle_binary'] = (predict_df['tackle_prob'] > 0.75).astype(int)

total_tackles = (predict_df.groupby(['gameId', 'playId', 'displayName'])['att_tackle']
                 .max()  # Takes the maximum value per group, which would be 1 if a tackle happened in the play
                 .reset_index()  # Flatten the multi-index into columns
                 .groupby('displayName')['att_tackle']
                 .sum())  # Sums up all the tackles per player across plays

# Calculate the mean tackle_prob for the first 10 frames of each play
mean_pred_tackles_first_10_frames = (predict_df[predict_df['frameId'] <= 15]
                                    .groupby(['gameId', 'playId', 'displayName'])['tackle_binary']
                                    .mean()  # Calculate the mean tackle_prob per group
                                    .reset_index()  # Flatten the multi-index into columns
                                    .groupby('displayName')['tackle_binary']
                                    .sum())  # Sum up the mean probabilities per player across plays

# Calculate the mean tackle_prob over the whole play and adjust by att_tackle
mean_tackle_prob_whole_play = (predict_df
                                    .groupby(['gameId', 'playId', 'displayName'])['tackle_binary']
                                    .mean()  # Calculate the mean tackle_prob per group
                                    .reset_index()  # Flatten the multi-index into columns
                                    .groupby('displayName')['tackle_binary']
                                    .sum())  # Sum up the mean probabilities per player across plays

# Calculate the differences between actual tackles and mean predicted tackles
difference_first_10_frames = total_tackles - mean_pred_tackles_first_10_frames
difference_whole_play = total_tackles - mean_tackle_prob_whole_play

# Combine the results into one dataframe for a comprehensive view
results_df = pd.DataFrame({
    'Total Tackles': total_tackles,
    'Mean Predicted Tackles First 10 Frames': mean_pred_tackles_first_10_frames,
    'Mean Predicted Tackles Whole Play': mean_tackle_prob_whole_play,
    'Difference First 10 Frames': difference_first_10_frames,
    'Difference Whole Play': difference_whole_play
}).fillna(0).reset_index()  # Fill NA values with 0

results_df.head()

"""
# Overall
Cross-validation recall scores: [0.72978304 0.67061144 0.71506058 0.68293886 0.70683947]
Mean CV recall: 0.7010
Recall on training data: 0.8939
Precision on training data: 0.2476
F1-Score on training data: 0.3877
ROC-AUC on training data: 0.7666

Recall on test data: 0.8326
Precision on test data: 0.2462
F1-Score on test data: 0.3800
ROC-AUC on test data: 0.7455

# First 5 Frames
Recall on test data: 0.7720
Precision on test data: 0.1869
F1-Score on test data: 0.3010
ROC-AUC on test data: 0.6809

# Last 5 Frames
Recall on test data: 0.9382
Precision on test data: 0.3634
F1-Score on test data: 0.5239
ROC-AUC on test data: 0.8632
"""
###################################### Randomized search for xgboost model ################

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
###############################################################################
bst2 = xgb.train(params_lloss, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=10, verbose_eval=True)

base_models = [
    ('xgb_auc', xgb.XGBClassifier(**params_auc)),
    ('xgb_logloss', xgb.XGBClassifier(**params_lloss))]

meta_model = LogisticRegression()

stacking_clf = StackingClassifier(estimators=base_models,final_estimator=meta_model, cv=None)

# Fit the stacking classifier on the training data
stacking_clf.fit(X_train, y_train)

# Evaluate on validation data
y_pred = stacking_clf.predict(X_test)

# You can then compute metrics on y_val_pred to evaluate the performance
recall_stacking = recall_score(y_test, y_pred)
print(f"Recall of stacked model: {recall_stacking:.4f}")