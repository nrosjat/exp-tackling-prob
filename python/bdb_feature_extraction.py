# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 08:58:18 2023

@author: Dr. Nils Rosjat
"""

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from bdb_functions import count_players, calculate_deviation, compute_speeds, compute_mean_distances, get_dist

project_dir = 'data'
os.listdir(project_dir)

# load player data 
players = pd.read_csv(f"{project_dir}/players.csv")


for w in range(9):
    # load data from one week 
    week = pd.read_csv(f"{project_dir}/tracking_week_{w+1}.csv")

    # join player positioning information onto a week's worth of tracking data 
    week = week.merge(players.loc[:, ['nflId', 'position']], how='left')


    # new dataframe for data 
    df = pd.DataFrame()
    for gid in week['gameId'].unique():
        # subset data down to one game
        game = week.loc[week['gameId'] == gid].copy()
        
        for pid in game['playId'].unique():
            # subset data down to one play
            play = game.loc[game['playId'] == pid].copy()
    
            for fid in play['frameId'].unique():
                # subset data down to one frame 
                frame = play.loc[play['frameId'] == fid].copy()
    
                # make unique positions, as to not duplicate columns based on player position
                frame['pos_unique'] = (frame['position']
                                    .add(frame
                                          .groupby('position', as_index=False)
                                          .cumcount()
                                          .add(1)
                                          .dropna()
                                          .astype(str)
                                          .str.replace('.0', '', regex=False)
                                          .str.replace('0', '', regex=False)))
    
                # calc distances 
                _df = (pd
                     .DataFrame(cdist(frame.loc[:, ['x', 'y']], 
                                      frame.loc[:, ['x', 'y']]), 
                                index=frame['nflId'], 
                                columns=frame['pos_unique'].fillna('football')))
    
                # reset index to pop out nflId into its own column
                _df = _df.reset_index()
    
                # merge new distance values onto original dataframe
                frame = frame.merge(_df)
    
                # concatenate new results into the output dataframe 
                df = pd.concat([df, frame])
    
    # Save 'df' to a file in the 'data' subfolder
    df.to_csv(f'data/tracking_dist_week_{w+1}.csv', index=False)

# load tracking data
tracking_datafile = [
                    "data/tracking_dist_week_1.csv",
                    "data/tracking_dist_week_2.csv",
                    "data/tracking_dist_week_3.csv",
                    "data/tracking_dist_week_4.csv",
                    "data/tracking_dist_week_5.csv",
                    "data/tracking_dist_week_6.csv",
                    "data/tracking_dist_week_7.csv",
                    "data/tracking_dist_week_8.csv",
                    "data/tracking_dist_week_9.csv"
                    ]

# Unify playing direction to right for all plays
for file in tracking_datafile:
    
    tracking_data = pd.read_csv(file)
    fname = file[:-4] + '_ori.csv'
    # Apply transformations based on playDirection
    mask = tracking_data['playDirection'] == 'left'
    
    tracking_data.loc[mask, 'x'] = 120 - tracking_data.loc[mask, 'x']
    tracking_data.loc[mask, 'y'] = (160 / 3) - tracking_data.loc[mask, 'y']
    tracking_data.loc[mask, 'o'] = 180 + tracking_data.loc[mask, 'o']
    tracking_data.loc[mask, 'o'] = tracking_data.loc[mask, 'o'].apply(lambda x: x - 360 if x > 360 else x)
    tracking_data.loc[mask, 'dir'] = 180 + tracking_data.loc[mask, 'dir']
    tracking_data.loc[mask, 'dir'] = tracking_data.loc[mask, 'dir'].apply(lambda x: x - 360 if x > 360 else x)
    
    tracking_data.to_csv(fname)
    
reoriented_tracking_datafile = [
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

for file in reoriented_tracking_datafile:
    
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