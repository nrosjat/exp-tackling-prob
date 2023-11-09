# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:11:24 2023

@author: Lagor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import pandas as pd
from bdb_functions import plot_field_with_heatmap_ani

def create_animation(gameId, playId, flip, save_path):
    # Load your data here (play_data, offense_data, defense_play, and predict_df)
    # fetch play data
    play_data = pd.read_csv("data/plays.csv")
    play = play_data[(play_data.gameId == gameId) & (play_data.playId == playId)]
    off = play.possessionTeam.values[0]

    # Read tracking data
    week7_data = pd.read_csv('data/tracking_week_7.csv')
    week8_data = pd.read_csv('data/tracking_week_8.csv')
    week9_data = pd.read_csv('data/tracking_week_9.csv')
    offense_data = pd.concat([week7_data, week8_data, week9_data])

    defense_play = predict_df[(predict_df.gameId == gameId) & (predict_df.playId == playId)]
    offense_play = offense_data[(offense_data.gameId == gameId) & (offense_data.playId == playId) & (offense_data.club == off)]
    football_play = offense_data[(offense_data.gameId == gameId) & (offense_data.playId == playId) & (offense_data.club == 'football')]

    # Create the main figure and subplots
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # fetch play data
    play_data = pd.read_csv("data/plays.csv")
    play = play_data[(play_data.gameId == gameId) & (play_data.playId == playId)]
    off = play.possessionTeam.values[0]

    # Read tracking data
    week7_data = pd.read_csv('data/tracking_week_7.csv')
    week8_data = pd.read_csv('data/tracking_week_8.csv')
    week9_data = pd.read_csv('data/tracking_week_9.csv')
    offense_data = pd.concat([week7_data, week8_data, week9_data])

    defense_play = predict_df[(predict_df.gameId == gameId) & (predict_df.playId == playId)]
    offense_play = offense_data[(offense_data.gameId == gameId) & (offense_data.playId == playId) & (offense_data.club == off)]
    football_play = offense_data[(offense_data.gameId == gameId) & (offense_data.playId == playId) & (offense_data.club == 'football')]

    line_chart_data = {}

    def update_line_chart(frame_id):
        # Extract data for the current frameId
        defense_data = defense_play[defense_play['frameId'] == frame_id].copy()

        # Update the line chart data
        for index, row in defense_data.iterrows():
            nflId = row['nflId']
            tackle_prob = row['tackle_prob']
            player_name = row['displayName']  # Get player name

            if nflId not in line_chart_data:
                line_chart_data[nflId] = {'x': [], 'y': [], 'name': player_name}  # Store player name

            line_chart_data[nflId]['x'].append((frame_id - 5) * 0.1)  # Convert frameId to time in seconds
            line_chart_data[nflId]['y'].append(tackle_prob)

        # Clear the line chart subplot
        ax2.clear()

        # Plot the line chart for each player with labels
        for nflId, data in line_chart_data.items():
            ax2.plot(data['x'], data['y'], label=data['name'])  # Use player name as label

        # Set labels and title for the line chart subplot
        ax2.set_xlabel('Time (in s)')
        ax2.set_ylabel('Tackle Probability')
        ax2.set_title('Tackle Probability for Players')

        # Add labels at the end of each line
        for nflId, data in line_chart_data.items():
            last_x, last_y = data['x'][-1], data['y'][-1]
            ax2.text(last_x, last_y, data['name'], fontsize=8, verticalalignment='bottom', horizontalalignment='left', rotation=0)

    # Extract all unique frameIds
    frame_ids = sorted(defense_play['frameId'].unique())

    # Calculate the maximum frameId for scaling the x-axis
    max_frame_id = max(frame_ids)

    # The update function for the animation
    def update(frame_id):
        ax1.clear()

        # Extract data for the current frameId
        defense_data = defense_play[defense_play['frameId'] == frame_id].copy()
        offense_data = offense_play[offense_play['frameId'] == frame_id].copy()
        football_data = football_play[football_play['frameId'] == frame_id].copy()

        if flip:
            # Flip if flip is true
            offense_data['x'] = 120 - offense_data['x']
            offense_data['y'] = 53.3 - offense_data['y']
            football_data['x'] = 120 - football_data['x']
            football_data['y'] = 53.3 - football_data['y']

        plot_field_with_heatmap_ani(defense_data['x'].values, defense_data['y'].values, defense_data['tackle_prob'].values, 
                                offense_data['x'].values, offense_data['y'].values, 
                                football_data['x'].values[0], football_data['y'].values[0], play, ax1)
        update_line_chart(frame_id)

        # Set the x-axis limits based on the maximum frameId
        ax2.set_xlim(0, (max_frame_id - 5) * 0.1)

        return ax1, ax2

    # Create the animation
    ani = FuncAnimation(fig, update, frames=frame_ids, repeat=False)

    # Save the animation as a GIF
    ani.save(save_path, writer='pillow', fps=2)

# Example usage:
create_animation(2022102306, 4135, False, 'game_animation.gif')
