# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:11:46 2023

@author: Dr. Nils Rosjat
"""
import pandas as pd

predict_df = pd.read_excel('predictions.xlsx')

total_tackles = (predict_df.groupby(['gameId', 'playId', 'displayName'])['att_tackle']
                 .max()  # Takes the maximum value per group, which would be 1 if a tackle happened in the play
                 .reset_index()  # Flatten the multi-index into columns
                 .groupby('displayName')['att_tackle']
                 .sum())  # Sums up all the tackles per player across plays


# Calculate the mean tackle_prob for the first 10 frames of each play
mean_pred_tackles_first_10_frames = (predict_df[predict_df['frameId'] <= 15]
                                    .groupby(['gameId', 'playId', 'displayName'])['tackle_binary']
                                    .mean()  # Calculate the mean tackle_prob per group
                                    .reset_index()  # Flatten the multi-index into columns
                                    .groupby('displayName')['tackle_binary']
                                    .sum())  # Sum up the mean probabilities per player across plays

# Calculate the mean tackle_prob over the whole play and adjust by att_tackle
mean_tackle_prob_whole_play = (predict_df
                                    .groupby(['gameId', 'playId', 'displayName'])['tackle_binary']
                                    .mean()  # Calculate the mean tackle_prob per group
                                    .reset_index()  # Flatten the multi-index into columns
                                    .groupby('displayName')['tackle_binary']
                                    .sum())  # Sum up the mean probabilities per player across plays

# Calculate the differences between actual tackles and mean predicted tackles
difference_first_10_frames = total_tackles - mean_pred_tackles_first_10_frames
difference_whole_play = total_tackles - mean_tackle_prob_whole_play


# Function to calculate mean probability for a range of frames
def mean_probability(df, frame_col, start_frame_offset, end_frame_offset):
    return df[(df['frameId'] >= df[frame_col] + start_frame_offset) & 
              (df['frameId'] <= df[frame_col] + end_frame_offset)] \
            .groupby(['gameId', 'playId', 'displayName'])['tackle_prob'] \
            .mean() \
            .reset_index()

# Calculate the mean tackle probability at the beginning of the play (using frames 5-10)
mean_start = mean_probability(predict_df, 'frameId', 0, 10)

# Calculate the last frame for each game and play
last_frame = predict_df.groupby(['gameId', 'playId'])['frameId'].max().reset_index()
last_frame.rename(columns={'frameId': 'last_frameId'}, inplace=True)

# Merge last_frame with predict_df
predict_df_merged = predict_df.merge(last_frame, on=['gameId', 'playId'])

# Calculate the mean tackle probability at the end of the play (using last 5 frames)
mean_end = mean_probability(predict_df_merged, 'last_frameId', -4, 0)

# Calculate the change in probability
prob_change = mean_end.set_index(['gameId', 'playId', 'displayName']) \
                      .subtract(mean_start.set_index(['gameId', 'playId', 'displayName']), fill_value=0)

# Sum the changes for each player across all plays and convert to Series
sum_changes_series = prob_change.groupby('displayName').sum()['tackle_prob']

# Rename the Series for clarity
sum_changes_series.name = 'Sum of Probability Changes'

# Count the number of plays each player appeared in
plays_count = predict_df.groupby('displayName').apply(lambda x: len(x[['gameId', 'playId']].drop_duplicates()))

# Combine the results into one dataframe for a comprehensive view
results_df = pd.DataFrame({
    'Number of Plays': plays_count,
    'Total Tackles': total_tackles,
    'Mean Predicted Tackles First 10 Frames': mean_pred_tackles_first_10_frames,
    'Mean Predicted Tackles Whole Play': mean_tackle_prob_whole_play,
    'Difference First 10 Frames': difference_first_10_frames,
    'Difference Whole Play': difference_whole_play,
    'Probability Increase': sum_changes_series
}).fillna(0).reset_index()  # Fill NA values with 0

results_df.to_excel('results_output.xlsx')