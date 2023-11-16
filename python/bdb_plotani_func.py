# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:11:24 2023

@author: Dr. Nils Rosjat
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import pandas as pd
from bdb_functions import plot_field_with_heatmap_ani
from sklearn.linear_model import LinearRegression
import seaborn as sns

def create_animation(gameId, playId, flip, save_path):
    """
    Creates an animation for a specified play in a football game.

    Parameters:
    gameId (int): The unique identifier for the game.
    playId (int): The unique identifier for the play within the game.
    flip (bool): Whether to flip the original play's tracking orientation in case of direction "left".
    save_path (str): The file path where the generated animation will be saved.

    This function first checks if a global dataframe 'predict_df' is defined. If not,
    it loads the data from 'predictions.xlsx'. It then processes play data and tracking data
    from specified CSV files, and creates an animation of the selected play. The animation
    is saved to the provided path.

    Example Usage:
    create_animation(2022102306, 4135, False, 'game_animation.gif')
    """
    
    # load predictions if not loaded already    
    try:
        predict_df
    except NameError:
        predict_df = pd.read_excel('predictions.xlsx')

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

def plot_tackle_comparison(df, choice, save_path):
    """
    Creates and saves a lollipop plot comparing actual and predicted tackles.

    Parameters:
    df (DataFrame): DataFrame containing the data.
    choice (str): Either 'Whole Play' or 'First 10 Frames'.
    save_path (str): Path where the PNG file will be saved.
    
    Example Usage:
    >>> results_df = [your dataframe here]
    >>> plot_tackle_comparison(df=results_df, choice='Whole Play', save_path='Output/actual_vs_predicted_tackles_top10.png')
    """

    # Choose the columns based on the specified choice
    if choice == 'Whole Play':
        diff_col = 'Difference Whole Play'
        mean_col = 'Mean Predicted Tackles Whole Play'
    elif choice == 'First 10 Frames':
        diff_col = 'Difference First 10 Frames'
        mean_col = 'Mean Predicted Tackles First 10 Frames'
    else:
        raise ValueError("Choice must be either 'Whole Play' or 'First 10 Frames'")

    # Sorting the DataFrame by the difference column and selecting the top 10
    top_10_players_diff = df.nlargest(10, diff_col)

    # Setting up the colormap
    norm = plt.Normalize(top_10_players_diff[diff_col].min(), top_10_players_diff[diff_col].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    # Setting up the horizontal plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Drawing lollipop lines and circles for each player with color scale
    for index, row in top_10_players_diff.iterrows():
        color = sm.to_rgba(row[diff_col])
        ax.plot([row['Total Tackles'], row[mean_col]], 
                [row['displayName'], row['displayName']], 
                color=color, marker='o', markersize=8, linewidth=2)

    # Adding labels, title, and colorbar
    ax.set_xlabel('Number of Tackles')
    title = f'Actual vs Predicted Tackles for Top 10 Players (Sorted by {diff_col})'
    ax.set_title(title)
    cbar = plt.colorbar(sm)
    cbar.set_label(diff_col)
    plt.grid(True)

    # Save the figure
    plt.savefig(save_path, format='png')
    plt.close()
    
def plot_scatter(df, x_variable, y_variable,  save_path, scale_variable=None):
    """
    Creates and saves a scatter plot comparing actual and predicted tackles or predicted tackles at different timepoints of the play.

    Parameters:
    df (DataFrame): DataFrame containing the data.
    x_variable, y_variable (str): Either 'Total Tackles', 'Mean Predicted Tackles Whole Play' or 'Mean Predicted Tackles First 10 Frames'.
    save_path (str): Path where the PNG file will be saved.
    
    Example Usage:
    >>> results_df = [your dataframe here]
    >>> plot_scatter(df=results_df, x_variable='Mean Predicted Tackles Whole Play', y_variable='Total Tackles', save_path='Output/scatter_predict_vs_actual.png')
    """
    
    plt.figure(figsize=(10, 6))
   
    if scale_variable:
        # Scale dot sizes by the number of actual tackles
        sizes = df[scale_variable]
        ax = sns.regplot(x=x_variable, y=y_variable, data=df, scatter_kws={'s': sizes}, line_kws={'color': 'red'})
    else:
        ax = sns.regplot(x=x_variable, y=y_variable, data=df, line_kws={'color': 'red'})
    
    plt.xlabel(x_variable)
    plt.ylabel(y_variable)
     
    # Save the figure
    plt.savefig(save_path, format='png')
    plt.close()