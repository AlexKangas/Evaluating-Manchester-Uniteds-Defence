
# -*- coding: utf-8 -*-
"""
@author: Alex Kangas

A large part of this code is adapted from code provided by David Sumpter and Aleksander Andrzejewski:
https://soccermatics.readthedocs.io/en/latest/
"""
import pandas as pd
import numpy as np
import json
# plotting
import matplotlib.pyplot as plt
# statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf
#opening data
import os
import pathlib
import warnings
#used for plots
from scipy import stats
from mplsoccer import PyPizza, FontManager
#used for plots
from mplsoccer import Pitch
from scipy.stats import binned_statistic_2d
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')
##############################################################################
# Opening data
# ----------------------------
train = pd.DataFrame()
file_name = 'events_England.json'
path = os.path.join(str(pathlib.Path().resolve().parents[0]), 'data', 'Wyscout',
file_name)
with open(path) as f:
data = json.load(f)
train = pd.concat([train, pd.DataFrame(data)])
#potential data collection error handling
train = train.loc[train.apply (lambda x: len(x.positions) == 2, axis = 1)]
##############################################################################
# Calculating air and defensive ground duels won
# ----------------------------
def wonDuels(df):
"""
Parameters
----------
df : dataframe
dataframe with Wyscout event data.
Returns
-------
duels_won: dataframe
dataframe with number of won air and defensive ground duels for a player
"""
#find air duels
air_duels = df.loc[df["subEventName"] == "Air duel"]
#703 is the id of a won duel
won_air_duels = air_duels.loc[air_duels.apply (lambda x:{'id':703} in x.tags,
axis = 1)]
#group and sum air duels
wad_player = won_air_duels.groupby(["playerId"]).eventId.count().reset_index()
wad_player.rename(columns = {'eventId':'air_duels_won'}, inplace=True)
#find ground duels won
ground_duels = df.loc[df["subEventName"].isin(["Ground defending duel"])]
won_ground_duels = ground_duels.loc[ground_duels.apply (lambda x:{'id':1801} in
x.tags, axis = 1)]
wgd_player =
won_ground_duels.groupby(["playerId"]).eventId.count().reset_index()
wgd_player.rename(columns = {'eventId':'ground_duels_won'}, inplace=True)
#outer join
duels_won = wgd_player.merge(wad_player, how = "outer", on = ["playerId"])
return duels_won
duels = wonDuels(train)
#investigate structure
duels.head(3)
##############################################################################
#Calculating interceptions
# ----------------------------
def calculateInterceptions(df):
"""
Parameters
----------
df : dataframe
dataframe with Wyscout event data.
Returns
-------
interceptions: dataframe
dataframe with number of interceptions of a player.
"""
df = df.copy()
interceptions = df.loc[df.apply(lambda x:{'id':1401} in x.tags, axis = 1)]
interceptions_player =
interceptions.groupby(['playerId']).eventId.count().reset_index()
interceptions_player.rename(columns = {'eventId':'interceptions'},
inplace=True)
return interceptions_player
interceptions = calculateInterceptions(train)
#investigate structure
interceptions.head(3)
##############################################################################
#Calculating xT defence
# ----------------------------
def calculatexTDefence(train):
##############################################################################
# Actions moving the ball
# -----------------------
df = train.copy()
next_event = df.shift(-1, fill_value=0)
df["nextEvent"] = next_event["subEventName"]
df["nextEventId"] = next_event["id"]
df["kickedOut"] = df.apply(lambda x: 1 if x.nextEvent == "Ball out of the
field" else 0, axis = 1)
#get move_df
move_df = df.loc[df['subEventName'].isin(['Simple pass', 'High pass', 'Head
pass', 'Smart pass', 'Cross'])]
#filtering out of the field
delete_passes = move_df.loc[move_df["kickedOut"] == 1]
move_df = move_df.drop(delete_passes.index)
#extract coordinates
move_df["x"] = move_df.positions.apply(lambda cell: (cell[0]['x']) * 105/100)
move_df["y"] = move_df.positions.apply(lambda cell: (100 - cell[0]['y']) *
68/100)
move_df["end_x"] = move_df.positions.apply(lambda cell: (cell[1]['x']) *
105/100)
move_df["end_y"] = move_df.positions.apply(lambda cell: (100 - cell[1]['y']) *
68/100)
#create 2D histogram of these
pitch = Pitch(line_color='black',pitch_type='custom', pitch_length=105,
pitch_width=68, line_zorder = 2)
move = pitch.bin_statistic(move_df.x, move_df.y, statistic='count', bins=(16,
12), normalize=False)
move_count = move["statistic"]
##############################################################################
# Shots
# ----------------------------
#get shot df
shot_df = df.loc[df['subEventName'] == "Shot"]
shot_df["x"] = shot_df.positions.apply(lambda cell: (cell[0]['x']) * 105/100)
shot_df["y"] = shot_df.positions.apply(lambda cell: (100 - cell[0]['y']) *
68/100)
#create 2D histogram of these
shot = pitch.bin_statistic(shot_df.x, shot_df.y, statistic='count', bins=(16,
12), normalize=False)
shot_count = shot["statistic"]
##############################################################################
# Goals
# ----------------------------
# To calculate the Expected Threat we need also goals. We filter them
# from the shots dataframe.
# We store the number of goal occurences in each bin in 2D array as well.
#get goal df
goal_df = shot_df.loc[shot_df.apply(lambda x: {'id':101} in x.tags, axis = 1)]
goal = pitch.bin_statistic(goal_df.x, goal_df.y, statistic='count', bins=(16,
12), normalize=False)
goal_count = goal["statistic"]
##############################################################################
# Move probability
# ----------------------------
# We now need to calculate the probability of each moving action. To do so, we
divide its number
# in each bin by the sum of moving actions and shots in that bin. Then, we plot
it.
move_probability = move_count/(move_count+shot_count)
##############################################################################
# Move probability
# ----------------
# We also need to calculate the probability of a shot in each area. Again, we
divide its number
# in each bin by the sum of moving actions and shots in that bin. Then plot it.
shot_probability = shot_count/(move_count+shot_count)
##############################################################################
# Goal probability
# ----------------
# The next thing needed is the goal probability. It's calculated here in a
# rather naive way - number of goals in this area divided by number of shots
there.
# This is a simplified expected goals model.
goal_probability = goal_count/shot_count
goal_probability[np.isnan(goal_probability)] = 0
##############################################################################
# Transition matirices
# --------------------
# For each of 192 sectors we need to calculate a transition matrix - a matrix
of probabilities
# going from one zone to another one given that the ball was moved. First, we
create
# another columns in the *move_df*
# with the bin on the histogram that the event started and ended in. Then, we
group the data
# by starting sector and count starts from each of them. As the next step, for
each of the sectors
# we calculate the probability of transfering the ball from it to all 192
sectors on the pitch.
# given that the ball was moved. We do it as the division of events that went
to the end sector
# by all events that started in the starting sector. As the last step, we
vizualize the
# transition matrix for the sector in the bottom left corner of the pitch.
#move start index - using the same function as mplsoccer, it should work
move_df["start_sector"] = move_df.apply(lambda row: tuple([i[0] for i in
binned_statistic_2d(np.ravel(row.x), np.ravel(row.y),
values = "None",
statistic="count",
bins=(16, 12),
range=[[0, 105], [0, 68]],
expand_binnumbers=True)[3]]), axis = 1)
#move end index
move_df["end_sector"] = move_df.apply(lambda row: tuple([i[0] for i in
binned_statistic_2d(np.ravel(row.end_x), np.ravel(row.end_y),
values = "None",
statistic="count",
bins=(16, 12),
range=[[0, 105], [0, 68]],
expand_binnumbers=True)[3]]), axis = 1)
#df with summed events from each index
df_count_starts = move_df.groupby(["start_sector"])
["eventId"].count().reset_index()
df_count_starts.rename(columns = {'eventId':'count_starts'}, inplace=True)
transition_matrices = []
for i, row in df_count_starts.iterrows():
start_sector = row['start_sector']
count_starts = row['count_starts']
#get all events that started in this sector
this_sector = move_df.loc[move_df["start_sector"] == start_sector]
df_cound_ends = this_sector.groupby(["end_sector"])
["eventId"].count().reset_index()
df_cound_ends.rename(columns = {'eventId':'count_ends'}, inplace=True)
T_matrix = np.zeros((12, 16))
for j, row2 in df_cound_ends.iterrows():
end_sector = row2["end_sector"]
value = row2["count_ends"]
T_matrix[end_sector[1] - 1][end_sector[0] - 1] = value
T_matrix = T_matrix / count_starts
transition_matrices.append(T_matrix)
##############################################################################
# Calculating Expected Threat matrix
# ----------------------------
# We are now ready to calculate the Expected Threat. We do it by first
calculating
# (probability of a shot)*(probability of a goal given a shot). This gives the
probability of a
# goal being scored right away. This is the shoot_expected_payoff. We then add
this to
# the move_expected_payoff, which is what the payoff (probability of a goal)
will be
# if the player passes the ball. It is this which is the xT
#
# By iterating this process 6 times, the xT gradually converges to its final
value.
transition_matrices_array = np.array(transition_matrices)
xT = np.zeros((12, 16))
for i in range(5):
shoot_expected_payoff = goal_probability*shot_probability
move_expected_payoff =
move_probability*(np.sum(np.sum(transition_matrices_array*xT, axis = 2), axis =
1).reshape(16,12).T)
xT = shoot_expected_payoff + move_expected_payoff
##############################################################################
# Applying xT value to moving actions
# -----------------------------------
next_event.drop(next_event.tail(1).index,inplace=True)
interceptions_df = next_event.loc[next_event.apply(lambda x:{'id':1401} in
x.tags, axis = 1)]
unsuccessful_moves = move_df.loc[move_df.apply(lambda x:{'id':1802} in x.tags,
axis = 1)]
unsuccessful_moves["xT_defense"] = unsuccessful_moves.apply(lambda row:
xT[row.start_sector[1] - 1][row.start_sector[0] - 1], axis = 1)
interceptions_df.rename(columns = {'id':'nextEventId'}, inplace=True)
to_merge = unsuccessful_moves[['nextEventId', 'xT_defense']]
df_interceptions_xT = interceptions_df.merge(to_merge, how="inner",
on=["nextEventId"])
player_defensive_xT =
df_interceptions_xT.groupby(['playerId']).sum().reset_index()
player_defensive_xT = player_defensive_xT[['playerId', 'xT_defense']]
return player_defensive_xT
player_defensive_xT = calculatexTDefence(train)
#investigate structure
player_defensive_xT.head(3)
##############################################################################
# Minutes played
# ----------------------------
# All data on our plot will be per 90 minutes played. Therefore, we need an
information on the number of minutes played
# throughout the season. To do so, we will use a prepared file that bases on the
idea developed by students
# taking part in course in 2021. Files with miutes per game for players in top 5
leagues can be found
# `here <https://github.com/soccermatics/Soccermatics/tree/main/course/lessons/
minutes_played>`_. After downloading data and saving
# it in out directory, we open it and store in a dataframe. Then we calculate the
sum of miutes played in a season for each player.
path = os.path.join(str(pathlib.Path().resolve().parents[0]),"minutes_played",
'minutes_played_per_game_England.json')
with open(path) as f:
minutes_per_game = json.load(f)
minutes_per_game = pd.DataFrame(minutes_per_game)
minutes = minutes_per_game.groupby(["playerId"]).minutesPlayed.sum().reset_index()
##############################################################################
# Summary table
# ----------------------------
# To make our radar plots we need to first prepare the data with previously
calculated statistics. We left join
# (too keep all the players). Also, we right join minutes, because there may be
some players who were on the pitch
# but didn't make an action. Then, the na observations are filled with zeros (if
there was NA scored goals it meant).
# Moreover, we filter out players who played 400 miutes or less.
players = train["playerId"].unique()
summary = pd.DataFrame(players, columns = ["playerId"])
summary = summary.merge(player_defensive_xT, how = "left", on =
["playerId"]).merge(interceptions, how = "left", on = ["playerId"]).merge(duels,
how = "left", on = ["playerId"])
summary = minutes.merge(summary, how = "left", on = ["playerId"])
summary = summary.fillna(0)
summary = summary.loc[summary["minutesPlayed"] > 400]
##############################################################################
# Filtering positions
# ----------------------------
# Since we would like to create a plot with attacking values, it is important to
keep only forwards (also the player that we will
# make the plot for is a forward). Therefore, we open the players dataset, we
filter out forwards and inner join it with our summary
# dataframe to keep only Premier League forwards who played more than 400 minutes.
path = os.path.join(str(pathlib.Path().resolve().parents[0]),"data", 'Wyscout',
'players.json')
player_df = pd.read_json(path, encoding='unicode-escape')
defenders = player_df.loc[player_df.apply(lambda x: x.role["name"] == "Defender",
axis = 1)]
defenders.rename(columns = {'wyId':'playerId'}, inplace=True)
to_merge = defenders[['playerId', 'shortName']]
summary = summary.merge(to_merge, how = "inner", on = ["playerId"])
##############################################################################
# Calculating statistics per 90
# ----------------------------
# To adjust the data for different number of minutes played, we calculate each
statistic we
# want to plot per 90 minutes player. That means that we multiply it by 90 and
divide by
# the total number of minutes played by player.
summary_per_90 = pd.DataFrame()
summary_per_90["shortName"] = summary["shortName"]
for column in summary.columns[2:-1]:
summary_per_90[column + "_per90"] = summary.apply(lambda x:
x[column]*90/x["minutesPlayed"], axis = 1)
##############################################################################
# Calculating possession
# ----------------------------
# As the next step we would like to adjust our plot by the player's team ball
possesion while they
# were on the pitch. To do it, for each row of our dataframe with minutes per
player per each game
# we take all the events that were made in this game while the player was on the
pitch.
# We will also use duels, but
# don't include lost air duels and lost ground defending duels. Why? Possesion is
calculated as number of touches by team divided
# by the number all touches. If a player lost ground defending duel, that means
that he could have been dribbled by, so he did not
# touch the ball. If they lost the air duel, they lost a header. Therefore, we
claim that those were mostly events where player may have not
# touched the ball (or if he did the team did not take control over it). We sum
# both team passes and these duels and all passes and these duels in this period.
We store these values in a
# dictionary. Then, summing them for each player separately and calculating their
ratio, we get
# the possesion of the ball by player's team while he was on the pitch. As the last
step we merge it with our summary dataframe.
possesion_dict = {}
#for every row in the dataframe
for i, row in minutes_per_game.iterrows():
#take player id, team id and match id, minute in and minute out
player_id, team_id, match_id = row["playerId"], row["teamId"], row["matchId"]
#create a key in dictionary if player encounterd first time
if not str(player_id) in possesion_dict.keys():
possesion_dict[str(player_id)] = {'team_passes': 0, 'all_passes' : 0}
min_in = row["player_in_min"]*60
min_out = row["player_out_min"]*60
#get the dataframe of events from the game
match_df = train.loc[train["matchId"] == match_id].copy()
#add to 2H the highest value of 1H
match_df.loc[match_df["matchPeriod"] == "2H", 'eventSec'] =
match_df.loc[match_df["matchPeriod"] == "2H", 'eventSec'] +
match_df.loc[match_df["matchPeriod"] == "1H"]["eventSec"].iloc[-1]
#take all events from this game and this period
player_in_match_df = match_df.loc[match_df["eventSec"] >
min_in].loc[match_df["eventSec"] <= min_out]
#take all passes and won duels as described
all_passes =
player_in_match_df.loc[player_in_match_df["eventName"].isin(["Pass", "Duel"])]
#adjusting for no passes in this period (Tuanzebe)
if len(all_passes) > 0:
#removing lost air duels
no_contact = all_passes.loc[all_passes["subEventName"].isin(["Air duel",
"Ground defending duel","Ground loose ball duel"])].loc[all_passes.apply(lambda x:
{'id':701} in x.tags, axis = 1)]
all_passes = all_passes.drop(no_contact.index)
#take team passes
team_passes = all_passes.loc[all_passes["teamId"] == team_id]
#append it {player id: {team passes: sum, all passes : sum}}
possesion_dict[str(player_id)]["team_passes"] += len(team_passes)
possesion_dict[str(player_id)]["all_passes"] += len(all_passes)
#calculate possesion for each player
percentage_dict = {key: value["team_passes"]/value["all_passes"] if
value["all_passes"] > 0 else 0 for key, value in possesion_dict.items()}
#create a dataframe
percentage_df = pd.DataFrame(percentage_dict.items(), columns = ["playerId",
"possesion"])
percentage_df["playerId"] = percentage_df["playerId"].astype(int)
#merge it
summary = summary.merge(percentage_df, how = "left", on = ["playerId"])
##############################################################################
# Adjusting data for possession
# ----------------------------
# Since we would like to adjust our values by possession, we divide the total
statistics by the
# possesion while player was on the pitch during the entire season. To normalize
the values per
# 90 minutes player we repeat the multiplication by 90 and division by minutes
played.
#create a new dataframe only for it
summary_adjusted = pd.DataFrame()
summary_adjusted["shortName"] = summary["shortName"]
#calculate value adjusted
for column in summary.columns[2:6]:
summary_adjusted[column + "_adjusted_per90"] = summary.apply(lambda x:
(x[column]/(1-x["possesion"]))*90/x["minutesPlayed"], axis = 1)
##############################################################################
# Making the plot with adjusted data for Manchester United defenders
# ----------------------------
# After calculating the values, we repeat the steps by calculating percentiles and
plotting radars from
# making the plot per 90. Note that this time we show the percentile rank on the
plot.
manutd_defenders = defenders.loc[defenders["currentTeamId"] == 1611]
for name in manutd_defenders["shortName"]:
player_adjusted = summary_adjusted.loc[summary_adjusted["shortName"] == name]
player_adjusted = player_adjusted[['xT_defense_adjusted_per90',
"interceptions_adjusted_per90", "ground_duels_won_adjusted_per90",
"air_duels_won_adjusted_per90"]]
#take only necessary columns
adjusted_columns = player_adjusted.columns[:]
#values
values = [player_adjusted[column].iloc[0] for column in adjusted_columns]
#percentiles
percentiles = [int(stats.percentileofscore(summary_adjusted[column],
player_adjusted[column].iloc[0])) for column in adjusted_columns]
names = ["xT_defense", "interceptions","Defensive Ground Duels Won", "Air Duels
Won"]
#list of names on plots
names = ["xT_defense", "interceptions","Defensive Ground Duels Won", "Air Duels
Won"]
slice_colors = ["blue"] * 2 + ["red"] * 2
text_colors = ["white"]*4
font_normal =
FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/"
"Roboto%5Bwdth,wght%5D.ttf?raw=true"))
font_bold =
FontManager(("https://github.com/google/fonts/blob/main/apache/robotoslab/"
"RobotoSlab%5Bwght%5D.ttf?raw=true"))
font_normal =
FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/"
"Roboto%5Bwdth,wght%5D.ttf?raw=true"))
font_italic =
FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/"
"Roboto-Italic%5Bwdth,wght%5D.ttf?raw=true"))
font_bold =
FontManager(("https://github.com/google/fonts/blob/main/apache/robotoslab/"
"RobotoSlab%5Bwght%5D.ttf?raw=true"))
baker = PyPizza(
params=names, # list of parameters
straight_line_color="#000000", # color for straight lines
straight_line_lw=1, # linewidth for straight lines
last_circle_lw=1, # linewidth of last circle
other_circle_lw=1, # linewidth for other circles
other_circle_ls="-." # linestyle for other circles
)
fig, ax = baker.make_pizza(
percentiles, # list of values
figsize=(10, 10), # adjust figsize according to your need
param_location=110,
slice_colors=slice_colors,
value_colors = text_colors,
value_bck_colors=slice_colors,
# where the parameters will be added
kwargs_slices=dict(
facecolor="cornflowerblue", edgecolor="#000000",
zorder=2, linewidth=1
), # values to be used when plotting slices
kwargs_params=dict(
color="#000000", fontsize=12,
fontproperties=font_normal.prop, va="center"
), # values to be used when adding parameter
kwargs_values=dict(
color="#000000", fontsize=12,
fontproperties=font_normal.prop, zorder=3,
bbox=dict(
