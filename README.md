# Analyzing Tackling Probabilities - Big Data Bowl 2024
Repo for the NFL Big Data Bowl 2024 competition

To use this repository, you need to have downloaded the 2024 NFL Big Data Bowl tracking data from [kaggle.com](https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/) and placed in a `data` subfolder.

The repository contains the following steps:

1. Data preprocessing `python/bdb_feature_extraction.py`

This script is strucutred in multiple subprocesses:

- Load tracking data, game data, play data and tackle data.
- Normalize plays to have direction 'right' by default (e.g. flip left directed plays by 180°).
- Extract valuable play-by-play information from play data and game data as well as information about tackler IDs for each play.
- Feature engineering from tacking data for the upcoming expected tackling probabilities model which include: distances between players of each play, relative speeds, number of defensive players closer to the ball carrier, number of offensive players closer to the ball carrier, mean distance to the 4 closest offensive players and mean distance to the 4 closest defensive players and the angular movement direction deviation from a direct path to the ball carrier.

(Several steps will create intermediate preprocessed data stored under `data` for convenience.)

2. Training expected tackling probabilities `python/bdb_model_training.py`

Train an XGBoost model based on the previously defined features to predict the probability of each defender to make a tackle at a given frame. The hyperparameters have been tuned by a `RandomizedSearchCV` that can be re-run by setting `randomsearch` to `True`. Prediction results paired with defender tracking data will be stored as `predictions.xlsx`.

3. Creating metrics from previously predicted data `python/bdb_analytics.py`

This function creates a output spreadsheet consisting of several entries:

- `displayName`: Defender Name
- `Total Tackles`: Actual total tackles (sum of solo and assisted tackles)
- `Mean Predicted Tackles First 10 Frames`: Number of predicted tackles defined from the mean probability throughout the first 10 frames (i.e. first second)
- `Mean Predicted Tackles Whole Play`: Number of predicted tackles defined from the mean probability throughout the whole play
- `Difference First 10 Frames`: Difference of actual tackles compared with predicted tackles first 10 frames
- `Difference Whole Play`: Difference of actual tackles compared with predicted tackles whole play
- `Probability Increase`: Sum of probability increase comparing the tackling probability in first 0.5 seconds with the tackling probability at the last 0.5 seconds (i.e. which player increased his tackle probability throughout the play)
  
4. Plotting functions `python/bdb_plotani_func.py`

Plotting functions for the graphics displayed in the kaggle notebook can be found here. Those functions are:

- `create_animation`: which creates a gif showing the players locations, tackle probability heatmaps alongside with a linechart of tackle probability development
- `plot_tackle_comparison`: which creates a lollipop chart comparing actual tackles with predicted tackles
- `plot_scatter`: which creates a scatter plot for comparison of actual tackles with predicted tackles as well as predicted tackles at different timepoints of the play
