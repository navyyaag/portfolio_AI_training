#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries

import pandas as pd
import datetime as dt
import scipy.stats as sp
import numpy as np
import statsmodels.formula.api as sm 


# In[3]:


Shotlog_1415=pd.read_csv("C:/Users/KLIN/Downloads/Shotlog_14_15.csv")
pd.set_option('display.max_columns',50)
Shotlog_1415


# In[32]:


Shotlog_1415['shoot_player'].unique()


# In[4]:


Shotlog_1415.columns.to_list()


# In[6]:


Shotlog_1415['date']=pd.to_datetime(Shotlog_1415['date'])
Shotlog_1415['date'].describe()


# In[7]:


Shotlog_1415=Shotlog_1415.sort_values(by=['shoot_player','date','quarter','game_clock','shot_clock'], ascending=[True,True,True,False,False])
Shotlog_1415


# In[8]:


Shotlog_1415['lag_shot_hit']=Shotlog_1415.groupby(['shoot_player','game_id'])['current_shot_hit'].shift(1)
Shotlog_1415.head(50)


# In[9]:


Shotlog_1415['error']=Shotlog_1415['current_shot_hit']-Shotlog_1415['average_hit']
Shotlog_1415['lagerror']=Shotlog_1415['lag_shot_hit']-Shotlog_1415['average_hit']


# In[25]:


reg1=sm.ols(formula='error~lagerror+shot_dist+dribbles+C(points)+C(quarter)+touch_time+home_away+shoot_player+closest_defender+closest_def_dist',data=Shotlog_1415).fit()
print(reg1.summary())


# In[26]:


reg2=sm.wls(formula='error~lagerror+shot_dist+dribbles+C(points)+C(quarter)+touch_time+home_away+shoot_player+closest_defender+closest_def_dist',weights=1/Shotlog_1415['shot_per_game'],data=Shotlog_1415).fit()
print(reg2.summary())


# In[22]:


def reg_player(player):
    shoot_player=Shotlog_1415[Shotlog_1415['shoot_player']==player]
    reg_player=sm.ols(formula='error~lagerror+shot_dist+dribbles+points+C(quarter)+touch_time+home_away+closest_def_dist',data=shoot_player).fit()
    print(reg_player.summary())
    return;


# In[23]:


reg_player('aaron brooks')


# In[24]:


def reg_wls_player(player):
    shoot_player=Shotlog_1415[Shotlog_1415['shoot_player']==player]
    reg_player=sm.wls(formula='error~lagerror+shot_dist+dribbles+points+C(quarter)+touch_time+home_away+closest_def_dist',weights=1/shoot_player['shot_per_game'],data=shoot_player).fit()
    print(reg_player.summary())
    return;


# In[27]:


reg_player('andrew wiggins')


# In[28]:


reg_player('stephen curry')


# In[35]:


reg_player('russell westbrook')


# In[30]:


reg_player('james harden')


# In[36]:


reg_wls_player('alonzo gee')


# In[37]:


reg_wls_player('cole aldrich')


# In[38]:


reg_wls_player('reggie jackson')


# In[39]:


reg_wls_player('stephen curry')


# In[ ]:




