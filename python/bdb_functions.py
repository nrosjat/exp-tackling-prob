# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:00:17 2023

@author: Dr. Nils Rosjat
"""

import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse

def plot_field_with_heatmap_ani(x_coords, y_coords, probabilities, x_coords_off, y_coords_off, football_x, football_y, play, ax):
    # gg_field function definition (with zorder adjustments)
    def gg_field(yardmin=0, yardmax=120, buffer=5, direction="horiz",
                 field_color="forestgreen", line_color="white",
                 sideline_color=None, endzone_color="darkgreen", play=play):
        
        # Field dimensions (units = yards)
        xmin, xmax = 0, 120
        ymin, ymax = 0, 53.33
        
        # Distance from sideline to hash marks in the middle (70 feet, 9 inches)
        hash_dist = (70 * 12 + 9) / 36
        
        # Yard lines locations (every 5 yards)
        yd_lines = np.arange(15, 106, 5)
        
        # Hash mark locations (left 1 yard line to right 1 yard line)
        yd_hash = np.arange(11, 110)
        
        # Field number size
        num_size = 6
        
        # Rotate field numbers with field direction
        angle_vec = [0, 180] if direction == "horiz" else [270, 90]
        num_adj = [-2, 1] if direction == "horiz" else [1, -1]
        
        # Add field background
        ax.add_patch(plt.Rectangle((xmin, ymin - buffer), xmax - xmin, ymax - ymin + 2 * buffer, fill=True, color=field_color, zorder=1))
        
        # Add end zones
        ax.add_patch(plt.Rectangle((xmin, ymin), 10, ymax - ymin, fill=True, color=endzone_color, zorder=1))
        ax.add_patch(plt.Rectangle((xmax - 10, ymin), 10, ymax - ymin, fill=True, color=endzone_color, zorder=1))
        
        # Add yardlines every 5 yards
        for y in yd_lines:
            ax.axvline(x=y, ymin=0, ymax=1, color=line_color, zorder=1)
        
        # Add thicker lines for endzones, midfield, and sidelines
        ax.axvline(x=0, ymin=0, ymax=1, color=line_color, linewidth=1.8, zorder=1)
        ax.axvline(x=10, ymin=0, ymax=1, color=line_color, linewidth=1.8, zorder=1)
        ax.axvline(x=60, ymin=0, ymax=1, color=line_color, linewidth=1.8, zorder=1)
        ax.axvline(x=110, ymin=0, ymax=1, color=line_color, linewidth=1.8, zorder=1)
        ax.axvline(x=120, ymin=0, ymax=1, color=line_color, linewidth=1.8, zorder=1)
        
        # Add a blue line at the line of scrimmage
        line_of_scrimmage = play.absoluteYardlineNumber.values[0]
        ax.axvline(x=line_of_scrimmage, color='blue', linewidth=2, zorder = 1)
        
        # Calculate the yardstogo
        yardstogo = play.yardsToGo.values[0]
        
        # Add a yellow target line
        target_line = line_of_scrimmage + yardstogo
        ax.axvline(x=target_line, color='yellow', linewidth=2, zorder= 1)
        
        ax.axhline(y=0, xmin=0, xmax=1, color=line_color, linewidth=1.8, zorder=1)
        ax.axhline(y=ymax, xmin=0, xmax=1, color=line_color, linewidth=1.8, zorder=1)
        
        # Numbers for both loops: 1,2,3,4,5,4,3,2,1
        numbers = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    
        # Add field numbers (every 10 yards)
        for y in np.arange(20, 101, 10):
            ax.text(y + num_adj[1], ymin + 12, "0", rotation=angle_vec[0], color=line_color, size=num_size)
        
        for y, num in zip(np.arange(20, 101, 10), numbers):
            ax.text(y + num_adj[0], ymin + 12, str(num), rotation=angle_vec[0], color=line_color, size=num_size)
        
        for y, num in zip(np.arange(20, 101, 10), numbers):
            ax.text(y + num_adj[1], ymax - 12, str(num), rotation=angle_vec[1], color=line_color, size=num_size)
        
        for y in np.arange(20, 101, 10):
            ax.text(y + num_adj[0], ymax - 12, "0", rotation=angle_vec[1], color=line_color, size=num_size)
        
        # Add hash marks - middle of the field
        for y in yd_hash:
            ax.plot([y, y], [hash_dist - 0.5, hash_dist + 0.5], color=line_color)
            ax.plot([y, y], [ymax - hash_dist - 0.5, ymax - hash_dist + 0.5], color=line_color)
        
        # Add hash marks - sidelines
        for y in yd_hash:
            ax.plot([y, y], [ymax, ymax - 1], color=line_color)
            ax.plot([y, y], [ymin, ymin + 1], color=line_color)
        
        # Add conversion lines at 2-yard line
        ax.plot([12, 12], [(ymax - 1) / 2, (ymax + 1) / 2], color=line_color)
        ax.plot([108, 108], [(ymax - 1) / 2, (ymax + 1) / 2], color=line_color)
        
        # Cover up lines outside of the field with sideline_color
        ax.add_patch(plt.Rectangle((xmin, ymax), xmax - xmin, buffer, fill=True, color=sideline_color, zorder=1))
        ax.add_patch(plt.Rectangle((xmin, ymin - buffer), xmax - xmin, buffer, fill=True, color=sideline_color, zorder=1))
        
        # Remove axis labels and tick marks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set axis limits based on field dimensions
        if direction == "horiz":
            ax.set_xlim(yardmin, yardmax)
            ax.set_ylim(ymin - buffer, ymax + buffer)
        elif direction == "vert":
            ax.set_xlim(ymin - buffer, ymax + buffer)
            ax.set_ylim(yardmin, yardmax)
            ax.invert_xaxis()  # Flip the plot for vertical orientation
            
        return ax

    # Generate a 2D grid of probabilities
    xx, yy = np.mgrid[0:120:1, 0:53:1]  # Assuming a field size of 120x53.33
    heatmap_data = np.zeros_like(xx, dtype=float)
    
    for x, y, prob in zip(x_coords, y_coords, probabilities):
        heatmap_data += prob * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * (2**2)))
        
    # Create a custom colormap
    blues = plt.cm.Blues
    transparent_index = 10
    cm = mcolors.LinearSegmentedColormap.from_list(
        "CustomBlues", [(1, 1, 1, 0)] * transparent_index + [blues(i) for i in range(transparent_index, 256)]
    )

    # Call the gg_field function
    ax = gg_field(yardmin=0, yardmax=120, buffer=5, direction="horiz", sideline_color="forestgreen", play=play)
    
    # Display the heatmap on the football field
    ax.imshow(heatmap_data.T, extent=[0, 120, 0, 53.33], origin='lower', alpha=0.7, cmap=cm, aspect='auto', zorder=2)

    # Overlay player locations
    ax.scatter(x_coords, y_coords, color='red', s=50, zorder=3)

    # Overlay offense locations
    ax.scatter(x_coords_off, y_coords_off, color='blue', s=50, zorder=3)
    
    # Add a brown ellipsoid to represent the football
    football = Ellipse((football_x, football_y), width=2, height=1, color="brown", zorder=4)
    ax.add_patch(football)
    
    # Add text above the plot
    description = play.playDescription.values[0]
    ax.text(60, 65, description, color='black', fontsize=12, ha='center', va='center')
    
    return ax

def plot_field_with_heatmap(x_coords, y_coords, probabilities, x_coords_off, y_coords_off, football_x, football_y):
    # gg_field function definition (with zorder adjustments)
    def gg_field(yardmin=0, yardmax=120, buffer=5, direction="horiz",
                 field_color="forestgreen", line_color="white",
                 sideline_color=None, endzone_color="darkgreen"):
        
        # Field dimensions (units = yards)
        xmin, xmax = 0, 120
        ymin, ymax = 0, 53.33
        
        # Distance from sideline to hash marks in the middle (70 feet, 9 inches)
        hash_dist = (70 * 12 + 9) / 36
        
        # Yard lines locations (every 5 yards)
        yd_lines = np.arange(15, 106, 5)
        
        # Hash mark locations (left 1 yard line to right 1 yard line)
        yd_hash = np.arange(11, 110)
        
        # Field number size
        num_size = 6
        
        # Rotate field numbers with field direction
        angle_vec = [0, 180] if direction == "horiz" else [270, 90]
        num_adj = [-2, 1] if direction == "horiz" else [1, -1]
        
        # Create Matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Add field background
        ax.add_patch(plt.Rectangle((xmin, ymin - buffer), xmax - xmin, ymax - ymin + 2 * buffer, fill=True, color=field_color, zorder=1))
        
        # Add end zones
        ax.add_patch(plt.Rectangle((xmin, ymin), 10, ymax - ymin, fill=True, color=endzone_color, zorder=1))
        ax.add_patch(plt.Rectangle((xmax - 10, ymin), 10, ymax - ymin, fill=True, color=endzone_color, zorder=1))
        
        # Add yardlines every 5 yards
        for y in yd_lines:
            ax.axvline(x=y, ymin=0, ymax=1, color=line_color, zorder=1)
        
        # Add thicker lines for endzones, midfield, and sidelines
        ax.axvline(x=0, ymin=0, ymax=1, color=line_color, linewidth=1.8, zorder=1)
        ax.axvline(x=10, ymin=0, ymax=1, color=line_color, linewidth=1.8, zorder=1)
        ax.axvline(x=60, ymin=0, ymax=1, color=line_color, linewidth=1.8, zorder=1)
        ax.axvline(x=110, ymin=0, ymax=1, color=line_color, linewidth=1.8, zorder=1)
        ax.axvline(x=120, ymin=0, ymax=1, color=line_color, linewidth=1.8, zorder=1)
        
        ax.axhline(y=0, xmin=0, xmax=1, color=line_color, linewidth=1.8, zorder=1)
        ax.axhline(y=ymax, xmin=0, xmax=1, color=line_color, linewidth=1.8, zorder=1)
        
        # Numbers for both loops: 1,2,3,4,5,4,3,2,1
        numbers = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    
        # Add field numbers (every 10 yards)
        for y in np.arange(20, 101, 10):
            ax.text(y + num_adj[1], ymin + 12, "0", rotation=angle_vec[0], color=line_color, size=num_size)
        
        for y, num in zip(np.arange(20, 101, 10), numbers):
            ax.text(y + num_adj[0], ymin + 12, str(num), rotation=angle_vec[0], color=line_color, size=num_size)
        
        for y, num in zip(np.arange(20, 101, 10), numbers):
            ax.text(y + num_adj[1], ymax - 12, str(num), rotation=angle_vec[1], color=line_color, size=num_size)
        
        for y in np.arange(20, 101, 10):
            ax.text(y + num_adj[0], ymax - 12, "0", rotation=angle_vec[1], color=line_color, size=num_size)
        
        # Add hash marks - middle of the field
        for y in yd_hash:
            ax.plot([y, y], [hash_dist - 0.5, hash_dist + 0.5], color=line_color)
            ax.plot([y, y], [ymax - hash_dist - 0.5, ymax - hash_dist + 0.5], color=line_color)
        
        # Add hash marks - sidelines
        for y in yd_hash:
            ax.plot([y, y], [ymax, ymax - 1], color=line_color)
            ax.plot([y, y], [ymin, ymin + 1], color=line_color)
        
        # Add conversion lines at 2-yard line
        ax.plot([12, 12], [(ymax - 1) / 2, (ymax + 1) / 2], color=line_color)
        ax.plot([108, 108], [(ymax - 1) / 2, (ymax + 1) / 2], color=line_color)
        
        # Cover up lines outside of the field with sideline_color
        ax.add_patch(plt.Rectangle((xmin, ymax), xmax - xmin, buffer, fill=True, color=sideline_color, zorder=1))
        ax.add_patch(plt.Rectangle((xmin, ymin - buffer), xmax - xmin, buffer, fill=True, color=sideline_color, zorder=1))
        
        # Remove axis labels and tick marks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set axis limits based on field dimensions
        if direction == "horiz":
            ax.set_xlim(yardmin, yardmax)
            ax.set_ylim(ymin - buffer, ymax + buffer)
        elif direction == "vert":
            ax.set_xlim(ymin - buffer, ymax + buffer)
            ax.set_ylim(yardmin, yardmax)
            ax.invert_xaxis()  # Flip the plot for vertical orientation
            
        return ax

    # Generate a 2D grid of probabilities
    xx, yy = np.mgrid[0:120:1, 0:53:1]  # Assuming a field size of 120x53.33
    heatmap_data = np.zeros_like(xx, dtype=float)
    
    for x, y, prob in zip(x_coords, y_coords, probabilities):
        heatmap_data += prob * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * (2**2)))
        
    # Create a custom colormap
    blues = plt.cm.Blues
    transparent_index = 10
    cm = mcolors.LinearSegmentedColormap.from_list(
        "CustomBlues", [(1, 1, 1, 0)] * transparent_index + [blues(i) for i in range(transparent_index, 256)]
    )

    # Call the gg_field function
    ax = gg_field(yardmin=0, yardmax=120, buffer=5, direction="horiz", sideline_color="forestgreen")
    
    # Display the heatmap on the football field
    ax.imshow(heatmap_data.T, extent=[0, 120, 0, 53.33], origin='lower', alpha=0.7, cmap=cm, aspect='auto', zorder=2)

    # Overlay player locations
    ax.scatter(x_coords, y_coords, color='red', s=50, zorder=3)

    # Overlay offense locations
    ax.scatter(x_coords_off, y_coords_off, color='blue', s=50, zorder=3)
    
    # Add a brown ellipsoid to represent the football
    football = Ellipse((football_x, football_y), width=2, height=1, color="brown", zorder=4)
    ax.add_patch(football)

    plt.show()

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
