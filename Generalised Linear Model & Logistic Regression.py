#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[2]:


#Import Data
NBA = pd.read_csv("C:/Users/KLIN/Downloads/NBA_Games2.csv")


# In[3]:


pd.set_option('display.max_columns',50)
NBA


# In[4]:


NBA_2017=NBA[(NBA.SEASON_ID==22017)&(NBA.GAME_ID<1000000000)]
NBA_2017


# In[5]:


NBA_2017.describe()


# In[6]:


NBA_2017.info()


# In[7]:


NBA_2017.columns


# In[8]:


pd.set_option('display.max_rows',100000)
NBA_2017=NBA_2017[['TEAM_NAME', 'TEAM_ID','SEASON_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'MATCHUP','WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'PLUS_MINUS', 'WIN']]
NBA_2017


# In[9]:


NBA_2017.rename(columns={'TEAM_ABBREVIATION':'TEAM1'},inplace=True)
NBA_2017['TEAM2']=NBA_2017['MATCHUP'].str[-3:]
NBA_2017


# In[10]:


NBA_2017=NBA_2017.sort_values('GAME_ID')


# In[11]:


NBA_2017


# In[12]:


NBA_2017['PTSAG']=NBA_2017['PTS']-NBA_2017['PLUS_MINUS']
NBA_2017


# In[13]:


NBA_2017['HOME']=np.where(NBA_2017['MATCHUP'].str[4]=='v',1,0)
NBA_2017


# In[14]:


NBA_2017_stats=NBA_2017.groupby('TEAM1')[['PTS','PTSAG','WIN']].sum()
NBA_2017_stats


# In[15]:


NBA_2017_count = NBA_2017.groupby('TEAM1').size().reset_index(name='GAME COUNT')
NBA_2017_stats=pd.merge(NBA_2017_stats,NBA_2017_count,on='TEAM1')
NBA_2017_stats.rename(columns={'TEAM1':'TEAM'},inplace=True)


# In[16]:


NBA_2017_stats


# In[17]:


NBA_2017_stats['wpc']=NBA_2017_stats['WIN']/NBA_2017_stats['GAME COUNT']
NBA_2017_stats['pyth']=(NBA_2017_stats['PTS']**2)/(NBA_2017_stats['PTS']**2+NBA_2017_stats['PTSAG']**2)
NBA_2017_stats


# In[18]:


NBA_2017['cumPTS']=NBA_2017.groupby('TEAM1')['PTS'].apply(lambda x: x.cumsum())
NBA_2017['cumPTSAG']=NBA_2017.groupby('TEAM1')['PTSAG'].apply(lambda x: x.cumsum())
NBA_2017


# In[19]:


NBA_2017['pyth']=(NBA_2017['cumPTS']**2)/(NBA_2017['cumPTS']**2+NBA_2017['cumPTSAG']**2)
NBA_2017


# In[20]:


lpm=smf.ols(formula='WIN~pyth',data=NBA_2017).fit()
print(lpm.summary())


# In[21]:


sns.regplot(x='pyth',y='WIN',marker='.',data=NBA_2017)


# In[34]:


logit=smf.glm(formula='WIN~pyth',data=NBA_2017, family=sm.families.Binomial())
logreg=logit.fit()
print(logreg.summary())


# In[31]:


x = NBA_2017['pyth'].to_numpy().reshape(-1,1)
y = NBA_2017['WIN'].to_numpy()
print("X shape:", x.shape)
print("y shape:", y.shape)


# In[32]:


x


# In[33]:


y


# In[35]:


logreg = sm.Logit(y,sm.add_constant(x)).fit()
logreg.summary()


# In[ ]:




