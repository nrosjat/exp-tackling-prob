# Analyzing Tackling Probabilities - Big Data Bowl 2024
Repo for the NFL Big Data Bowl 2024 competition

To use this repository, you need to have downloaded the 2024 NFL Big Data Bowl tracking data from [kaggle.com](https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/).

The repository contains the following steps:

1. Data preprocessing python/script.py

This script is strucutred in multiple subprocesses:

- Load tracking data, game data, play data and tackle data.
- Extract valuable play-by-play information from play data and game data as well as information about tackler IDs for each play.
- Feature engineering from tacking data for the upcoming expected tackling probabilities model which include: distances between players of each play, relative speeds, number of defensive players closer to the ball carrier, number of offensive players closer to the ball carrier, mean distance to the 4 closest offensive players and mean distance to the 4 closest defensive players and the angular movement direction deviation from a direct path to the ball carrier.

2. Training expected tackling probabilities

3. Predict play-level and player-level tackling probabilities