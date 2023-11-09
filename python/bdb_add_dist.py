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