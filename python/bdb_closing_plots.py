# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:07:01 2023

@author: Lagor
"""

import seaborn as sns
import matplotlib.pyplot as plt

# Exclude the last 5 frames for each playId
test = test.groupby('playId').apply(lambda x: x.head(-5)).reset_index(drop=True)

# Set up the figure and axes
plt.figure(figsize=(12, 8))

# Loop through each unique combination of playId and nflId
for (play, player), group in test.groupby(['playId', 'displayName']):
    sns.lineplot(x=group['frameId'], y=group['dist_ballCarrier'], label=f"Play {play}, Player {player}")

plt.title('Distance to Ball Carrier for Each Play and Player')
plt.xlabel('Frame ID')
plt.ylabel('Distance to Ball Carrier')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Frames of interest
frames_of_interest = [1, 10, 20, 40, 60]

# Set up the figure
plt.figure(figsize=(12, 8))

# Loop through frames of interest and plot distribution
for frame in frames_of_interest:
    sns.kdeplot(test[test['frameId'] == frame]['dist_ballCarrier'], label=f'Frame {frame}', shade=True)

plt.title('Distribution of Distances for Specific Frames')
plt.xlabel('Distance to Ball Carrier')
plt.ylabel('Density')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

############ Animate Predictions

offense_data = pd.read_csv('data/tracking_week_8.csv')
offense_data = offense_data[(offense_data.gameId == 2022103010) & (offense_data.playId == 1443) & (offense_data.frameId > 4) & (offense_data.club == 'SF')]

plot_game = predict_df[predict_df.gameId == 2022103010]
plot_play = plot_game[plot_game.playId == 1443]
plot_frame = plot_play[plot_play.frameId == 5]

# Extracting x, y coordinates and probabilities
x_coords = plot_frame['x'].values
y_coords = plot_frame['y'].values
probabilities = plot_frame['tackle_prob'].values

# Generate a 2D grid of probabilities
xx, yy = np.mgrid[0:100:1, 0:50:1]  # Assuming a field size of 100x50
heatmap_data = np.zeros_like(xx, dtype=float)

for x, y, prob in zip(x_coords, y_coords, probabilities):
    heatmap_data += prob * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * (2**2)))

# Plotting the heatmap
sns.heatmap(heatmap_data.T, cmap="Blues")

# Overlay player locations
plt.scatter(x_coords, y_coords, color='red', s=50)

# Display the plot
plt.title("Probability Heatmap of Players Making the Tackle")
plt.xlabel("X Coordinate on the Field")
plt.ylabel("Y Coordinate on the Field")
#plt.colorbar(label="Probability Density")
plt.show()

########################## Animation

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(10, 5))

def update(frame, offense_data):
    ax.clear()
       
    x_coords = frame['x'].values
    y_coords = frame['y'].values
    probabilities = frame['tackle_prob'].values

    xx, yy = np.mgrid[0:100:1, 0:50:1]
    heatmap_data = np.zeros_like(xx, dtype=float)

    for x, y, prob in zip(x_coords, y_coords, probabilities):
        heatmap_data += prob * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * (5**2)))

    sns.heatmap(heatmap_data.T, cmap="Blues", ax=ax, cbar=False, vmin=0, vmax=1)
    ax.scatter(x_coords, y_coords, color='red', s=50)
    
    # Offense scatter plot
    offense_x_coords = offense_frame_data['x'].values
    offense_y_coords = offense_frame_data['y'].values
    ax.scatter(offense_x_coords, offense_y_coords, color='green', s=50, label='Offense')


grouped_frames = [group for _, group in plot_play.groupby('frameId')]
grouped_offense = [group for _, group in offense_data.groupby('frameId')]
ani = FuncAnimation(fig, update, frames=zip(grouped_frames, grouped_offense), repeat=True)
plt.show()