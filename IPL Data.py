#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import json


# In[2]:


pd.set_option('max_columns',30)


# In[3]:


def extract_deliveries(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    info = data['info']
    innings = data['innings']

    deliveries = []

    for i, inn in enumerate(innings, 1):
        for over_num, over in enumerate(inn['overs'], 1):
            for delivery_num, delivery in enumerate(over['deliveries'], 1):
                batter = delivery['batter']
                batting_team = inn['team']
                bowler = delivery['bowler']
                non_striker = delivery['non_striker']
                runs = delivery['runs']['total']
                extras = delivery['runs']['extras'] if 'extras' in delivery['runs'] else 0
                wide = extras['wides'] if isinstance(extras, dict) and 'wides' in extras else 0
                no_ball = extras['noballs'] if isinstance(extras, dict) and 'noballs' in extras else 0
                bye = extras['byes'] if isinstance(extras, dict) and 'byes' in extras else 0
                leg_bye = extras['legbyes'] if isinstance(extras, dict) and 'legbyes' in extras else 0
                wicket = None
                wicket_type = None
                wicket_fielders = None
                wicket_fielder_name = None
                if 'wickets' in delivery:
                    wicket = delivery['wickets'][0]['player_out']
                    wicket_type = delivery['wickets'][0]['kind']
                    wicket_fielders = delivery['wickets'][0]['fielders'] if 'fielders' in delivery['wickets'][0] else None
                    if wicket_fielders:
                        wicket_fielder_name = wicket_fielders[0]['name']
                    else:
                        wicket_fielder_name = None

                delivery_data ={
                    'inn': i,
                    'over': over_num,
                    'delivery': delivery_num,
                    'batter': batter,
                    'bowler': bowler,
                    'non_striker': non_striker,
                    'runs': runs,
                    'extras': extras,
                    'wide': wide,
                    'no_ball': no_ball,
                    'bye': bye,
                    'leg_bye': leg_bye,
                    'wicket': wicket,
                    'wicket_fielder_name': wicket_fielder_name,
                    'wicket_type': wicket_type,
                    'batting_team': batting_team,
                    'bowling_team': info['teams'][0] if batting_team==info['teams'][1] else info['teams'][1],
                    'stadium': info['venue'],
                    'match_date': info['dates'][0],
                }

                deliveries.append(delivery_data)

    df = pd.DataFrame(deliveries)

    return df


# In[15]:


file_path_df=pd.read_excel("C:/Users/KLIN/OneDrive/Desktop/IPL Data.xlsx",sheet_name='Sheet1')


# In[16]:


file_path_df


# In[17]:


result_dataframes = {}

for index, row in file_path_df.iterrows():
    file_path = row['FilePath']
    game_code = row['GameCode']
    
    dataframe_name = f"df_{game_code}"
    
    result_dataframes[dataframe_name] = extract_deliveries(file_path)


# In[18]:


processed_dataframes = {}

for game_code, dataframe in result_dataframes.items():
    
    #calculations
    dataframe['fours'] = np.where(dataframe['runs'] == 4, 1, 0)
    dataframe['sixes'] = np.where(dataframe['runs'] == 6, 1, 0)
    dataframe['dots'] = np.where(dataframe['runs'] == 0, 1, 0)
    wicket = dataframe.loc[pd.notnull(dataframe['wicket'])]

    # Batting stats
    wicket_batting = wicket.copy()
    wicket_batting['out'] = 1
    wicket_batting.rename(columns={'wicket': 'player'}, inplace=True)
    batting_stats = dataframe.groupby('batter').agg({'runs': 'sum', 'fours': 'sum', 'sixes': 'sum', 'dots': 'sum', 'delivery': 'count', 'batting_team': 'first', 'bowling_team': 'first', 'stadium': 'first', 'match_date': 'first'}).reset_index()
    batting_stats.rename(columns={'batter': 'player'}, inplace=True)
    batting_stats = pd.merge(batting_stats, wicket_batting[['player', 'out', 'bowler']], on='player', how='left')
    batting_stats.rename(columns={'runs': 'runs_batting', 'dots': 'dots_batting', 'delivery': 'delivery_batting', 'out': 'batter_out', 'batting_team': 'team', 'bowling_team': 'team_against'}, inplace=True)

    # Bowling stats
    bowling_stats = dataframe.groupby('bowler').agg({'runs': 'sum', 'extras': 'sum', 'wide': 'sum', 'no_ball': 'sum', 'bye': 'sum', 'leg_bye': 'sum', 'dots': 'sum', 'delivery': 'count', 'batting_team': 'first', 'bowling_team': 'first', 'stadium': 'first', 'match_date': 'first'}).reset_index()
    bowling_stats.rename(columns={'bowler': 'player'}, inplace=True)
    overs = dataframe.groupby(['bowler', 'over']).agg({'runs': 'sum'}).reset_index()
    overs['maiden_over'] = np.where(overs['runs'] == 0, 1, 0)
    overs = overs.groupby(['bowler']).agg({'maiden_over': 'sum'}).reset_index()
    overs.rename(columns={'bowler': 'player'}, inplace=True)
    wicket_bowling = wicket.copy()
    wicket_bowling.rename(columns={'bowler': 'player'}, inplace=True)
    wicket_bowling['wicket_excluding ro'] = np.where(wicket_bowling['wicket_type'] == 'run out', 0, 1)
    wicket_bowling = wicket_bowling.groupby('player').agg({'wicket_excluding ro': 'sum'}).reset_index()
    bowling_stats = pd.merge(bowling_stats, overs[['player', 'maiden_over']], on='player', how='left')
    bowling_stats = pd.merge(bowling_stats, wicket_bowling[['player', 'wicket_excluding ro']], on='player', how='left')
    bowling_stats.rename(columns={'runs': 'runs_bowling', 'dots': 'dots_bowling', 'delivery': 'delivery_bowling', 'batting_team': 'team_against', 'bowling_team': 'team'}, inplace=True)

    # Fielding stats
    wicket_fielding = wicket.loc[pd.notnull(wicket['wicket_fielder_name'])].copy()
    fielding_stats = wicket_fielding.groupby('wicket_fielder_name').agg({'wicket': 'count', 'stadium': 'first', 'match_date': 'first', 'batting_team': 'first', 'bowling_team': 'first'}).reset_index()
    fielding_stats.rename(columns={'wicket_fielder_name': 'player', 'wicket': 'catches', 'batting_team': 'team_against', 'bowling_team': 'team'}, inplace=True)

    # Aggregated player stats
    player_stats = pd.merge(bowling_stats, fielding_stats, on=['match_date', 'stadium', 'player', 'team', 'team_against'], how='outer')
    player_stats = pd.merge(batting_stats, player_stats, on=['match_date', 'stadium', 'player', 'team', 'team_against'], how='outer')

    # Store the resulting DataFrame in processed_dataframes using GameCode as key
    processed_dataframes[game_code] = player_stats


# In[19]:


for game_code, dataframe in result_dataframes.items():
    print('File code:', game_code)


# In[20]:


#processed_dataframes["df_{gamecode}"]


# In[21]:


#processed_dataframes["df_{gamecode}"].player.nunique()


# **Batting Points**
# 
# Run: +1
# 
# Boundary Bonus: +1
# 
# Six Bonus: +2
# 
# Half-Century Bonus: +8
# 
# Century Bonus: +16
# 
# Dismissal for a duck (Batter, Wicket-Keeper, & All-Rounder): -2

# **Bowling Points**
# 
# Wicket (Excluding Run Out): +25
# 
# 4 Wicket Bonus: +8
# 
# 5 Wicket Bonus: +16
# 
# Maiden Over: +8

# **Fielding Stats**
# 
# Catch: +8
# 
# Stumping/ Run Out (direct): +12 **NA**
# 
# Run Out (Thrower/Catcher): +6/6 **NA**

# In[22]:


database = pd.concat(processed_dataframes.values(), ignore_index=True)


# In[23]:


database


# In[24]:


database.to_excel("C:/Users/KLIN/OneDrive/Desktop/Database_IPL.xlsx", index=False)


# In[25]:


#player number verify (more than 22 due to fielding substitutes)
database.groupby(['match_date','stadium']).agg({'player':'count'})


# In[ ]:




