import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

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
        
        # Create Matplotlib figure and axis
#        fig, ax = plt.subplots(figsize=(10, 5))
        
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


# Initialize the figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

gameId = 2022102306
playId = 4135

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

# Extract all unique frameIds
frame_ids = sorted(defense_play['frameId'].unique())

# The update function for the animation
def update(frame_id):
    ax.clear()
    
    # Extract data for the current frameId
    defense_data = defense_play[defense_play['frameId'] == frame_id].copy()
    offense_data = offense_play[offense_play['frameId'] == frame_id].copy()
    football_data = football_play[football_play['frameId'] == frame_id].copy()
    

        # Flip the x and y coordinates for offense_data and football_data
       # offense_data['x'] = 120 - offense_data['x']
       # offense_data['y'] = 53.3 - offense_data['y']
       # football_data['x'] = 120 - football_data['x']
       # football_data['y'] = 53.3 - football_data['y']
    
    plot_field_with_heatmap_ani(defense_data['x'].values, defense_data['y'].values, defense_data['tackle_prob'].values, 
                            offense_data['x'].values, offense_data['y'].values, 
                            football_data['x'].values[0], football_data['y'].values[0],play,ax)
    return ax,

# Create the animation
ani = FuncAnimation(fig, update, frames=frame_ids, repeat=False)
# Save using ImageMagick writer
#plt.show()
f = r"animation4.gif"
writergif = animation.PillowWriter(fps=5) 
ani.save(f, writer=writergif)

f_mp4 = "animation4.mp4"
writer_mp4 = animation.FFMpegWriter(fps=5)  # 'libx264' codec for MP4
ani.save(f_mp4, writer=writer_mp4)

# game 2022100903 # play 1645
# highest jumps: 2022102306, 4135
