# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:25:11 2023

@author: Lagor
"""
import numpy as np
import pandas as pd

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

def compute_speeds(week):
    # Compute speed components
    week['vx'] = week['s'] * np.cos(np.deg2rad(week['dir']))
    week['vy'] = week['s'] * np.sin(np.deg2rad(week['dir']))

    # Create unique positions without innermost loop
    week['pos_unique'] = (week['position']
                          .add(week
                               .groupby(['gameId', 'playId', 'position'])
                               .cumcount()
                               .add(1)))

    # Compute pairwise differences in speed components using broadcasting
    vx = week['vx'].values[:, np.newaxis] - week['vx'].values
    vy = week['vy'].values[:, np.newaxis] - week['vy'].values

    # Compute the magnitude of these relative speeds
    relative_speeds = np.sqrt(vx**2 + vy**2)

    # Create a DataFrame for the relative speeds
    speeds_df = pd.DataFrame(relative_speeds, index=week['nflId'], columns=week['pos_unique'].fillna('football'))

    # Reset index to pop out nflId into its own column
    speeds_df = speeds_df.reset_index()

    # Merge new speed values onto the original dataframe
    week = week.merge(speeds_df, on='nflId', suffixes=('', '_speed'))

    return week
# Function to get the value from the column corresponding to ballCarrier's pos_unique value
def get_dist(row):
    gameId, playId = row['gameId'], row['playId']
    ballCarrierId = ballCarrier_map.get((gameId, playId), None)
    if ballCarrierId is None:
        return None
    ballCarrier_pos = tracking_df.loc[(tracking_df['gameId'] == gameId) & 
                                       (tracking_df['playId'] == playId) & 
                                       (tracking_df['nflId'] == ballCarrierId), 'pos_unique'].iloc[0]
    return row[ballCarrier_pos]

# Adjusted Step 3: Count offensive and defensive players closer to the ball within the same game, play, and frame
def count_players(row):
    # Filter for the current frame, game, and play
    current_data = tracking_df[(tracking_df['frameId'] == row['frameId']) & 
                               (tracking_df['gameId'] == row['gameId']) & 
                               (tracking_df['playId'] == row['playId'])]
    
    # Count offensive players closer to the ball
    offensive_count = sum((current_data['club'] == row['possessionTeam']) & (current_data['dist_ballCarrier'] < row['dist_ballCarrier']))
    
    # Count defensive players closer to the ball
    defensive_count = sum((current_data['club'] != row['possessionTeam']) & (current_data['club'] != 'football') & (current_data['dist_ballCarrier'] < row['dist_ballCarrier']))
    
    return pd.Series([offensive_count, defensive_count], index=['offensive_players_closer', 'defensive_players_closer'])

# Adjusted compute_mean_distances function
def compute_mean_distances(row):
    # If the row represents 'football', return 0 for both mean distances
    if row['club'] == 'football':
        return pd.Series([0, 0], index=['mean_dist_to_offense', 'mean_dist_to_defense'])

    # Filter for the current frame, game, and play
    current_data = tracking_df[(tracking_df['frameId'] == row['frameId']) & 
                               (tracking_df['gameId'] == row['gameId']) & 
                               (tracking_df['playId'] == row['playId']) & 
                               (tracking_df['club'] != 'football')]
    
    is_offensive = row['club'] == row['possessionTeam']
    
    # Calculate distances for offensive players
    offensive_players = current_data[current_data['club'] == row['possessionTeam']]
    offensive_distances = offensive_players.apply(lambda x: x[row['pos_unique']], axis=1)
    if is_offensive:
        mean_offensive_distance = np.mean(offensive_distances.nsmallest(5).iloc[1:])
    else:
        mean_offensive_distance = np.mean(offensive_distances.nsmallest(4))
    
    # Calculate distances for defensive players
    defensive_players = current_data[(current_data['club'] != row['possessionTeam']) & (current_data['club'] != 'football')]
    defensive_distances = defensive_players.apply(lambda x: x[row['pos_unique']], axis=1)
    if not is_offensive:
        mean_defensive_distance = np.mean(defensive_distances.nsmallest(5).iloc[1:])
    else:
        mean_defensive_distance = np.mean(defensive_distances.nsmallest(4))
    
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
    fname = file[:-4] + '_feature.csv'
    # Step 1: Merge the DataFrames on gameId, playId, and nflId
    tracking_df = pd.merge(tracking_data, 
                         tacklePlay_data[['gameId', 'playId', 'nflId', 'tackle', 'assist', 'forcedFumble', 'pff_missedTackle', 'att_tackle', 'possessionTeam']], 
                         on=['gameId', 'playId', 'nflId'], 
                         how='left')
    
    # Step 1: Replace NaN in possessionTeam
    most_freq_possession = tracking_df.groupby('playId')['possessionTeam'].apply(lambda x: x.mode()[0])
    tracking_df['possessionTeam'] = tracking_df.groupby('playId')['possessionTeam'].transform(lambda x: x.fillna(most_freq_possession[x.name]))

    print('Finished possession team.')    

    # Step 2: Add a new column dist_ballCarrier
    # Mapping of (gameId, playId) to ballCarrierId
    ballCarrier_map = tacklePlay_data.set_index(['gameId', 'playId'])['ballCarrierId'].to_dict()
    
    
    # Add the dist_ballCarrier column to the merged DataFrame
    tracking_df['dist_ballCarrier'] = tracking_df.apply(get_dist, axis=1)
    
    print('Finished ball carrier distance.')    
    
    
    # Step 2: Replace NaN in tackle-related columns
    cols_to_replace_nan = ['tackle', 'assist', 'forcedFumble', 'pff_missedTackle', 'att_tackle']
    tracking_df[cols_to_replace_nan] = tracking_df[cols_to_replace_nan].fillna(0)
    print('Finished replace nan for tackles.')    
    
    
    tracking_df[['offensive_players_closer', 'defensive_players_closer']] = tracking_df.apply(count_players, axis=1)
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