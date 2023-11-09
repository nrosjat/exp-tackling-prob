# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:34:40 2023

@author: Lagor
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 01:27:14 2023

@author: Lagor
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm

def calculate_deviation(x1, y1, x2, y2, player_direction_deg):
    # Calculate the angle of the direct path in radians
    direct_angle_rad = np.arctan2(y2 - y1, x2 - x1)
    
    # Convert player's direction to radians
    player_direction_rad = np.deg2rad(player_direction_deg)
    
    # Calculate the deviation in radians
    deviation_rad = np.arctan2(np.sin(direct_angle_rad - player_direction_rad), np.cos(direct_angle_rad - player_direction_rad))
    
    # Convert deviation to degrees
    deviation_deg = np.rad2deg(deviation_rad)
    
    return deviation_deg

project_dir = 'data'
os.listdir(project_dir)

# load player data 
players = pd.read_csv(f"{project_dir}/players.csv")


for file in tracking_datafile:
    # load data from one week 
    week = pd.read_csv(file,nrows=100)

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
                
                # Compute pairwise speed differences directly
                speeds = frame['s'].values
                relative_speeds_directional = speeds[:, None] - speeds
                
                # Create a DataFrame for the relative speeds
                speeds_df = pd.DataFrame(relative_speeds_directional, 
                                         index=frame['nflId'], 
                                         columns=frame['pos_unique'].fillna('football'))

                # reset index to pop out nflId into its own column
                speeds_df = speeds_df.reset_index()

                # merge new speed values onto the original dataframe
                frame = frame.merge(speeds_df, on='nflId', suffixes=('', '_speed'))
    
    # Save 'df' to a file in the 'data' subfolder
   # df.to_csv(f'data/tracking_dist_week_{w+1}.csv', index=False)