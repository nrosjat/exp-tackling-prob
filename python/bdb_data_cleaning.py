# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:45:20 2023

@author: Nils Rosjat
"""

import matplotlib.pyplot as plt # basic plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# fetch play data
play_data = pd.read_csv("data/plays.csv")

# fetch tackle data
tackle_data = pd.read_csv("data/tackles.csv")
tackle_data['att_tackle'] = tackle_data['tackle'] + tackle_data['assist']# + tackle_data['pff_missedTackle']

# fetch player data
player_data = pd.read_csv("data/players.csv")
player_data.drop(columns=['weight','birthDate','collegeName'],inplace=True)

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

columns_to_keep = ['gameId','playId','ballCarrierId','quarter','down','yardsToGo','passResult','playResult','absoluteYardlineNumber','offenseFormation','defendersInTheBox','homeTeamAbbr','visitorTeamAbbr','score_diff','time_rem_qtr','time_rem_half','time_rem_game']
gamePlay_data = gamePlay_data[columns_to_keep]

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

# merge tracking data with play data
play_tracking_dataset = []
for file in tracking_datafile:
    data = pd.read_csv(file)
    # select rows relevent with the route of the ball
    data = data.loc[data['displayName'] == 'football'] # remove for full data
    # join gamePlay_data and tracking_data on gameId and playId
    play_tracking_data = pd.merge(
    gamePlay_data,
    data,
    how="inner",
    left_on=["gameId","playId"],
    right_on=["gameId","playId"],
    sort=True,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    validate=None,
    )
    play_tracking_dataset.append(play_tracking_data)

# merge data tables into one dataset
play_tracking_df = pd.concat(play_tracking_dataset)

# create unique id
play_tracking_df['uniqueId'] = play_tracking_df['gameId'].astype(str) + play_tracking_df['playId'].astype(str)

ball_example = ball_route_right.loc[ball_route_right['uniqueId'] == '20220911012365']
# Sample plot
plt.plot(ball_example['x'], ball_example['y'], 'o')  # 'o' creates a scatter plot

# Annotate each point with the event text
for index, row in ball_example.iterrows():
    if not pd.isna(row['event']):
        plt.annotate(row['event'], (row['x'], row['y']))

plt.show()