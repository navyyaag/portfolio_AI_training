#!/usr/bin/env python
# coding: utf-8

# In[43]:


def get_game_data(gameid=1):
    '''Retrieves individual game data and creates a DataFrame of the home and away teams and their
    score results. This function assumes we are interested in the season 2017 and that we want
    only regular season games.

    :param gameid: The game number to retrieve data from.
    '''
    # We can pull down the JSON data directly from the NHL API
    game_url=f'https://statsapi.web.nhl.com/api/v1/game/201702{str(gameid).zfill(4)}/feed/live'
    http = urllib3.PoolManager()
    r = http.request('GET', game_url)
    data=json.loads(r.data)

    # The JSON data is pretty rich. For this analysis we want to get information on scoring which
    # is in the goals JSON object
    results=data['liveData']['plays']['currentPlay']['about']['goals']

    # We also need to get information on the home and away team names. This isn't useful for our
    # model per se, since we want to predict just whether the home team or the away team will win,
    # but we need this in order to connect to our other data sources
    teams={'home_team': data['gameData']['teams']['home']['name'], 'away_team': data['gameData']['teams']['away']['name']}

    # And we'll include when the game happened
    time={'time': data['metaData']['timeStamp']}

    # Now we can just bring these three dictionaries together. This might be unfamiliar syntax, 
    # it's called dictionary unpacking, but it just breaks each dictionary up and creates a new
    # dictionary which combines them all. In the end we want to work with pandas DataFrame objects
    # so that's what we can return to the caller (indexed by the time of the game)
    row={**results,**teams,**time}
    return pd.DataFrame(row, index=[row["time"]])

# Commented out for Coursera, you can uncomment the code below if you are running locally.
# Now let's just call this function for every game in the season, 1 through 1,271
game_results=pd.concat( [get_game_data(x) for x in range(1,12)] )

# And now that we've pulled this down, I'm going to save it for offline use
# game_results.to_csv("assets/game_results.csv")

# And let's take a look at that DataFrame
game_results


# In[3]:


gameid=1
str(gameid).zfill(4)


# In[4]:


game_url=f'https://statsapi.web.nhl.com/api/v1/game/201702{str(gameid).zfill(4)}/feed/live'


# In[5]:


game_url


# In[6]:


import urllib3
import json
import itertools

# And the standard data science data manipulation imports
import pandas as pd
import numpy as np


# In[7]:


game_url=f'https://statsapi.web.nhl.com/api/v1/game/201702{str(gameid).zfill(4)}/feed/live'
http = urllib3.PoolManager()
r = http.request('GET', game_url)
data=json.loads(r.data)


# In[8]:


pd.DataFrame(data)


# In[9]:


data['liveData']['plays']


# In[10]:


results=data['liveData']['plays']['currentPlay']['about']['goals']


# In[11]:


teams={'home_team': data['gameData']['teams']['home']['name'], 'away_team': data['gameData']['teams']['away']['name']}
teams


# In[12]:


data['gameData']['teams']['home']['name']


# In[13]:


time={'time': data['metaData']['timeStamp']}
time


# In[14]:


row={**results,**teams,**time}
pd.DataFrame(row, index=[row["time"]])


# In[15]:


row


# In[16]:


# Commented out for Coursera, you can uncomment the code below if you are running locally.

# Now, let's bring in salary information. I'm going to pull this down from a website called
# cap friendly. This website does not have an API, so we need to scrape it. Thankfully,
# pandas has a function which aims to turn HTML tables into DataFrames for us automatically
# called read_html(). The result of this function is a list of DataFrames, and I've manually
# inspected this to see that there is only one which has all of our cap information.
#salary=pd.read_html("https://www.capfriendly.com/archive/2017")[0]

# Now this website has pretty values of dollars, but we just want these as numeric values,
# so I'm going to change our column of interest (the final cap hit) to be stripped of
# commas and dollar signs
#salary["Final Cap Hit"]=salary["Final Cap Hit"].str.replace(',', '').str.replace('$', '').astype(int)

# Let's store this data to a file too
#salary.to_csv("assets/salary.csv",index=False)
#salary.head()


# In[17]:


# Great, we have two data sources down and ready for analysis, now we need to get some prior
# information about teams from the previous season. This will be useful for our model when
# we want to make early predictions and don't have the current season data.

# The NHL API has another great place to get standings for a whole season, so we'll use that
def team_standings(season="20162017"):
    '''Pull down the standings for teams in a single season.
    :param season: The season code (e.g. 20162017 for the 2016-2017 season)
    '''
    # Pull down the JSON data from the API directly
    game_url=f"https://statsapi.web.nhl.com/api/v1/standings?season={season}"
    http = urllib3.PoolManager()
    r = http.request('GET', game_url)
    data=json.loads(r.data)

    # In this case the JSON data has a record element for divisions and then lists the team 
    # records inside of that, so we need to do a nested iteration
    df_standings=pd.DataFrame()
    for record in data["records"]:
        for team_record in record["teamRecords"]:

            # We have to decide which standings we want to incorporate. Do we want just the
            # rank of the team from last season? The number of games they won? The number of
            # goals scored? This is where your knowledge of the sport can come in to add
            # context and value. I'm going to just include everything - for now - but this
            # is usually a poor choice in practice.

            # Since this is a JSON structure, and we want to turn it into a DataFrame, we can
            # use the handy json_normalize() function in pandas to "flatten" the JSON. And
            # we can just add that DataFrame to the bottom of our df_standings
            df_standings=df_standings.append(pd.json_normalize(team_record))
    return df_standings

# Commented out for Coursera, you can uncomment the code below if you are running locally.
previous_season_standings=team_standings()

# Let's save this for offline use
# previous_season_standings.to_csv("assets/previous_season_standings.csv",index=False)
previous_season_standings.head()


# In[18]:


game_url=f"https://statsapi.web.nhl.com/api/v1/standings?season=20162017"
http = urllib3.PoolManager()
r = http.request('GET', game_url)
data=json.loads(r.data)


# In[19]:


data


# In[20]:


record = data["records"][0]
record


# In[21]:


record["teamRecords"]


# In[54]:


df_cum=pd.DataFrame()
df_cum.loc['won', list(game_results["home_team"].unique()) ]=0
df_cum.loc['lost', list(game_results["home_team"].unique()) ]=0
df_cum.loc['won', list(game_results["away_team"].unique()) ]=0
df_cum.loc['lost', list(game_results["away_team"].unique()) ]=0


# In[55]:


df_cum


# In[56]:


df_cum=df_cum.unstack()
df_cum


# In[57]:


df_cum=pd.DataFrame(df_cum,columns=['time']).T
df_cum.head()


# In[58]:


for idx,row in game_results.iterrows():

    if row["away"]>row["home"]:
        winner=row["away_team"]
        loser=row["home_team"]
    elif row["away"]< row["home"]:
        winner=row["home_team"]
        loser=row["away_team"]

    df_cum.loc[idx, (winner,"won")]=df_cum[(winner,"won")].max()+1
    df_cum.loc[idx, (loser,"lost")]=df_cum[(loser,"lost")].max()+1

df_cum


# In[59]:


for idx,row in game_results.iterrows():

    if row["away"]>row["home"]:
        winner=row["away_team"]
        loser=row["home_team"]
    elif row["away"]< row["home"]:
        winner=row["home_team"]
        loser=row["away_team"]


# In[60]:


winner


# In[61]:


df_cum[(winner,"won")].max()


# In[62]:


df = pd.DataFrame({'A':[1,2,3,4,np.nan],'B':[5,6,7,np.nan,9]})
def g(df):
    return df.fillna(method='ffill')

result = g(df.copy())
print(result)


# In[65]:


df_cum=df_cum.fillna(method='ffill').drop(index="time")
df_cum.head()


# In[ ]:


def create_features(row):
    '''Operates on a single row of data from game_results, and interacts with global
    dataframes salary, previous_season_standings, and df_cum to generate a feature
    vector for that row.
    :param row: A single row in game_results
    :param return: A feature vector as a pandas Series object
    '''
    # Inside of this function let's store our features in a dictionary
    features={}

    # We can start by looking up the number of games the home and away teams have lost thus
    # far in the season
    features["away_won"]=df_cum.loc[row.name,(row["away_team"],"won")]
    features["away_lost"]=df_cum.loc[row.name,(row["away_team"],"lost")]
    features["home_won"]=df_cum.loc[row.name,(row["home_team"],"won")]
    features["home_lost"]=df_cum.loc[row.name,(row["home_team"],"lost")]

    # We have to adjust this to ensure that we're not leaking the results of this match!
    if row["outcome_categorical"]=="home":
        features["home_won"]=features["home_won"]-1
        features["away_lost"]=features["away_lost"]-1
    else:
        features["home_lost"]=features["home_lost"]-1
        features["away_won"]=features["away_won"]-1


    # Let's add in the salary cap information from last year
    features["away_cap"]=salary[row["away_team"]]
    features["home_cap"]=salary[row["home_team"]]

    # Let's get the previous season standings for each team too, and add an indicator
    # to each standing whether it was for the home or away team
    home_last_season=previous_season_standings.query(f"`team.name`=='{row['home_team']}'").add_prefix("home_last_season_")
    away_last_season=previous_season_standings.query(f"`team.name`=='{row['away_team']}'").add_prefix("away_last_season_")

    # Remember those Vegas Golden Knights? They didn't exist in the previous season, so
    # our code to convert the values to a dictionary won't work. We need to be robust to
    # this case, so let's just create an empty dictionary for teams which have no previous
    # season
    if len(home_last_season)>0:
        home_last_season=home_last_season.iloc[0].to_dict()
    else:
        home_last_season={}
    if len(away_last_season)>0:
        away_last_season=away_last_season.iloc[0].to_dict()
    else:
        away_last_season={}

    # Now we can leverage dictionary unpacking, returning all of the items including the
    # data from the game_results (which has our target variable) as a new Series
    return pd.Series({**features, **home_last_season, **away_last_season, **row})

# Let's generate these game results and put them into a new DataFrame called observations
observations=game_results.apply(create_features, axis='columns')
observations.head()

