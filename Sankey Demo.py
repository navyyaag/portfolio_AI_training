#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np


# In[7]:


projects = pd.read_excel("C:/Users/KLIN/Downloads/[PII] Compass Extract 18102023.xlsx", sheet_name='Projects',index_col='idProject')


# In[8]:


projects


# In[9]:


pd.set_option('display.max_columns',50)
projects


# In[10]:


projects.columns


# In[11]:


projects_cleaned = projects[['title', 'clientContactName', 'clientContactEmail',                              'description','idBranch', 'dateStart', 'dateEnd', 'statusSubmitted',                              'payment', 'revenue', 'revenueAmount', 'revenueCurrency',                             'feedbackRequested', 'statusDetails', 'statusAgreement','statusPayment',                             'statusFeedback', 'statusDeliverables', 'invoiceStatus', 'invoiceAmount',                             'invoiceCurrency', 'teamLeader', 'teamLeaderEmail', 'clientType',                              'projectType', 'projectLanguage']]


# In[12]:


projects_cleaned


# In[13]:


np.set_printoptions(threshold = np.inf)
print(projects_cleaned.title.unique())


# In[14]:


projects_cleaned.info()


# In[15]:


branches = pd.read_excel("C:/Users/KLIN/Downloads/[PII] Compass Extract 18102023.xlsx", sheet_name='Branches')


# In[16]:


branches.columns


# In[17]:


branches


# In[18]:


branches_cleaned=branches[['idBranch', 'nameFull', 'country', 'region', 'language', 'applicationsOpen', 'active', 'removed']]


# In[19]:


projects_cleaned = projects_cleaned.drop(projects_cleaned[pd.isna(projects_cleaned['revenueAmount']) & pd.isna(projects_cleaned['invoiceAmount'])].index)
projects_cleaned


# In[20]:


projects_cleaned.dropna(subset=['title'], inplace=True)


# In[21]:


projects_cleaned


# In[22]:


projects_cleaned.title.to_list()


# In[23]:


projects_cleaned['title'].astype(str)


# In[24]:


projects_cleaned['title'].describe()


# In[25]:


filtered_df = projects_cleaned[projects_cleaned['title'].str.contains('test',case = False, na=False)]
df1= projects_cleaned[projects_cleaned['title'].str.contains('Example',case = False, na=False)]
df2= projects_cleaned[projects_cleaned['title'].str.contains('Sample',case = False, na=False)]
filtered_df = pd.concat([filtered_df, df1])
filtered_df = pd.concat([filtered_df, df2])
filtered_df.drop([368],inplace=True)


# In[26]:


filtered_df


# In[27]:


projects_cleaned.drop(filtered_df.index,inplace=True)


# In[28]:


projects_cleaned


# In[29]:


remove = projects_cleaned[projects_cleaned['idBranch']==7]
remove


# In[30]:


projects_cleaned.drop(remove.index,inplace=True)
projects_cleaned


# In[31]:


database = pd.merge(projects_cleaned,branches_cleaned,how="left", on='idBranch')


# In[32]:


database


# In[33]:


import datetime as dt


# In[34]:


database['dateStart']=pd.to_datetime(database['dateStart'],errors='coerce')
database['dateEnd']=pd.to_datetime(database['dateEnd'],errors='coerce')


# In[35]:


database['project_duration']=database['dateEnd']-database['dateStart']


# In[36]:


database.info()


# In[37]:


database.loc[database['project_duration'] < pd.Timedelta(0), 'project_duration']=pd.NaT


# In[38]:


database.describe()


# In[39]:


database[pd.isna(database.revenueAmount)]


# In[40]:


database.invoiceCurrency.describe()


# In[41]:


database.info()


# In[42]:


#database.to_excel("C:/Users/KLIN/Downloads/data_glt.xlsx",sheet_name='Data_Consolidated')


# In[43]:


code1=pd.get_dummies(database['statusDetails'])
code1


# In[44]:


code2=pd.get_dummies(database['statusAgreement'])
code2.loc[code2['warning']==1,'success']='x'
code2


# In[45]:


code3=pd.get_dummies(database['statusPayment'])
code3.loc[code3['warning']==1,'success']='x'
code3


# In[46]:


code4=pd.get_dummies(database['statusFeedback'])
code4


# In[47]:


code5=pd.get_dummies(database['statusDeliverables'])
code5


# In[48]:


code1.rename(columns={'success':'Details'},inplace=True)
code1.drop(['danger'],axis=1,inplace=True)

code2.rename(columns={'success':'Agreement'},inplace=True)
code2.drop(['warning','danger'],axis=1,inplace=True)

code3.rename(columns={'success':'Payment'},inplace=True)
code3.drop(['warning','danger'],axis=1,inplace=True)

code4.rename(columns={'success':'Feedback'},inplace=True)
code4.drop(['danger'],axis=1,inplace=True)

code5.rename(columns={'success':'Deliverables'},inplace=True)
code5.drop(['danger'],axis=1,inplace=True)


# In[49]:


codes=pd.concat([code1,code2,code3,code4,code5],axis=1)
codes


# In[50]:


codes['code']=codes['Details'].astype(str)+codes['Agreement'].astype(str)+codes['Payment'].astype(str)+codes['Feedback'].astype(str)+codes['Deliverables'].astype(str)


# In[51]:


codes


# In[52]:


database=pd.merge(database,codes['code'],on=database.index)


# In[53]:


database


# In[54]:


database.drop(['key_0'],axis=1,inplace=True)
database.head()


# In[55]:


database.drop(['clientContactName','clientContactEmail','feedbackRequested','teamLeader','teamLeaderEmail','applicationsOpen','active','removed'],axis=1,inplace=True)
database.info()


# In[56]:


pd.set_option('display.max_rows',300)
database.loc[pd.isna(database['revenueAmount'])&pd.isna(database['revenue'])]


# In[57]:


database['final_revenue']=database['revenueAmount']


# In[58]:


database.loc[pd.isna(database['revenueAmount'])&pd.isna(database['revenue']),'final_revenue']=database['invoiceAmount']


# In[59]:


database.info()


# In[60]:


database.loc[database['invoiceAmount']>database['revenueAmount']]


# In[61]:


database.loc[database['invoiceAmount']>database['revenueAmount'],'final_revenue']=database['invoiceAmount']


# In[62]:


database.info()


# In[63]:


pd.set_option('display.max_rows',1400)
database.sort_values(['final_revenue'])


# In[64]:


database[database['country']=='Spain']


# In[65]:


data_clean=database[database['final_revenue']>=20]
data_clean


# In[66]:


data_clean.shape


# In[67]:


data_clean.info()


# In[68]:


data_clean['country'].unique()


# In[69]:


data_clean.loc[data_clean['country']=='India ','country']='India'
data_clean.loc[data_clean['country']=='United States ','country']='United States'
data_clean.loc[data_clean['country']=='France ','country']='France'
data_clean.loc[data_clean['country']=='United Kingdom ','country']='United Kingdom'


# In[70]:


data_clean.loc[(data_clean['country']=='New Zealand')|(data_clean['country']=='Australia')               |(data_clean['country']=='Philippines')|(data_clean['country']=='Hong Kong')|              (data_clean['country']=='Indonesia')|(data_clean['country']=='Singapore')|              (data_clean['country']=='Malaysia')|(data_clean['country']=='Taiwan')|               (data_clean['country']=='India'),'region']='APAC'

data_clean.loc[(data_clean['country']=='Belgium')|(data_clean['country']=='Denmark')               |(data_clean['country']=='Italy')|(data_clean['country']=='Germany')|              (data_clean['country']=='Portugal')|(data_clean['country']=='Sweden')|              (data_clean['country']=='Spain')|(data_clean['country']=='Austria')|               (data_clean['country']=='Bulgaria')|(data_clean['country']=='France')|               (data_clean['country']=='Netherlands')|(data_clean['country']=='United Kingdom'),'region']='EMEA'

data_clean.loc[(data_clean['country']=='Mexico')|(data_clean['country']=='United States')               |(data_clean['country']=='Canada')|(data_clean['country']=='Peru')|              (data_clean['country']=='Brazil'),'region']='AMER'


# In[71]:


data_clean.info()


# In[72]:


data_clean['final_revenueCurrency']=data_clean['invoiceCurrency']
data_clean.loc[pd.isna(data_clean['final_revenueCurrency'])&(data_clean['revenueCurrency']==1),'final_revenueCurrency']='USD'
data_clean.loc[pd.isna(data_clean['final_revenueCurrency'])&(data_clean['revenueCurrency']==2),'final_revenueCurrency']='AUD'
data_clean.loc[pd.isna(data_clean['final_revenueCurrency'])&(data_clean['revenueCurrency']==3),'final_revenueCurrency']='EUR'
data_clean.loc[pd.isna(data_clean['final_revenueCurrency'])&(data_clean['revenueCurrency']==4),'final_revenueCurrency']='GBP'
data_clean.loc[pd.isna(data_clean['final_revenueCurrency'])&(data_clean['revenueCurrency']==6),'final_revenueCurrency']='NZD'
data_clean.loc[pd.isna(data_clean['final_revenueCurrency'])&(data_clean['revenueCurrency']==7),'final_revenueCurrency']='SEK'
data_clean.loc[pd.isna(data_clean['final_revenueCurrency'])&(data_clean['revenueCurrency']==8),'final_revenueCurrency']='INR'
data_clean.loc[pd.isna(data_clean['final_revenueCurrency'])&(data_clean['revenueCurrency']==9),'final_revenueCurrency']='MXN'
data_clean.loc[pd.isna(data_clean['final_revenueCurrency'])&(data_clean['revenueCurrency']==10),'final_revenueCurrency']='HKD'
data_clean.loc[pd.isna(data_clean['final_revenueCurrency'])&(data_clean['revenueCurrency']==11),'final_revenueCurrency']='PHP'


# In[73]:


data_clean.info()


# In[74]:


data_analysis=data_clean
data_analysis.drop(['revenue','revenueAmount','revenueCurrency','invoiceAmount','invoiceCurrency','invoiceStatus'],axis=1,inplace=True)
data_analysis.info()


# In[75]:


data_analysis


# In[76]:


data_clean.to_excel("C:/Users/KLIN/Downloads/data_glt2.xlsx",sheet_name='Data_Cleaned')
data_analysis.to_excel("C:/Users/KLIN/Downloads/data_glt3.xlsx",sheet_name='Data_Analysis')


# In[77]:


summary = data_analysis.groupby('final_revenueCurrency').agg({'final_revenue': ['mean','median', 'count']})
summary


# In[78]:


summary['final_revenue']['count']


# In[79]:


np.array(summary['final_revenue']['count'])[0]


# In[80]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
g = sns.FacetGrid(data_analysis, col="final_revenueCurrency", col_wrap=4, sharex=False, sharey=False)

g.map(plt.hist, 'final_revenue', bins=20)
g.map(plt.axvline,x=1000,linestyle='--',color='r')
g.map(plt.axvline, x=data_analysis['final_revenue'].median(), color='black')

g.set_axis_labels("final_revenue", "Count")
g.set_titles(col_template="{col_name} Currency")

plt.show()


# In[81]:


g.savefig("C:/Users/KLIN/Downloads/RevenueDist.png")


# In[82]:


data_analysis.describe()


# In[83]:


data_analysis.groupby('country')['final_revenue'].agg(['mean', 'count'])


# In[84]:


data_analysis.groupby('nameFull')['final_revenue'].agg(['mean', 'count'])


# In[85]:


data_analysis.groupby('clientType')['final_revenue'].agg(['mean', 'count'])


# In[86]:


data_analysis.groupby('projectType')['final_revenue'].agg(['mean', 'count'])


# In[87]:


data_analysis.groupby(['country','final_revenueCurrency'])['final_revenue'].agg(['mean', 'count'])


# In[88]:


data_analysis['normFinal_revenue']=0
data_analysis.loc[data_analysis['final_revenueCurrency']=='AUD','normFinal_revenue']=data_analysis['final_revenue']
data_analysis.loc[data_analysis['final_revenueCurrency']=='USD','normFinal_revenue']=data_analysis['final_revenue']*1.564555
data_analysis.loc[data_analysis['final_revenueCurrency']=='EUR','normFinal_revenue']=data_analysis['final_revenue']*1.65336
data_analysis.loc[data_analysis['final_revenueCurrency']=='GBP','normFinal_revenue']=data_analysis['final_revenue']*1.900989084
data_analysis.loc[data_analysis['final_revenueCurrency']=='NZD','normFinal_revenue']=data_analysis['final_revenue']*0.91529
data_analysis.loc[data_analysis['final_revenueCurrency']=='SEK','normFinal_revenue']=data_analysis['final_revenue']*0.1399539
data_analysis.loc[data_analysis['final_revenueCurrency']=='INR','normFinal_revenue']=data_analysis['final_revenue']*0.01879449855
data_analysis.loc[data_analysis['final_revenueCurrency']=='MXN','normFinal_revenue']=data_analysis['final_revenue']*0.0878465
data_analysis.loc[data_analysis['final_revenueCurrency']=='HKD','normFinal_revenue']=data_analysis['final_revenue']*0.199992
data_analysis.loc[data_analysis['final_revenueCurrency']=='PHP','normFinal_revenue']=data_analysis['final_revenue']*0.0275758764


# In[89]:


#data_analysis['normFinal_revenue']=data_analysis['normFinal_revenue'].astype(int)


# In[90]:


data_analysis


# In[91]:


data_analysis.info()


# In[92]:


data_analysis['statusSubmitted']=data_analysis['statusSubmitted'].astype(str)
data_analysis['idBranch']=data_analysis['idBranch'].astype(str)


# In[93]:


data_analysis.info()


# In[94]:


countrydata = data_analysis.groupby('country')['normFinal_revenue'].agg(['mean','sum', 'count'])
countrydata


# In[95]:


data_analysis['successCounter']=(data_analysis['statusSubmitted'].astype(int)+np.where(data_analysis['statusDetails']=='success',1,0)+                                np.where(data_analysis['statusAgreement']=='success',1,0)+np.where(data_analysis['statusPayment']=='success',1,0)+                                np.where(data_analysis['statusFeedback']=='success',1,0)+np.where(data_analysis['statusDeliverables']=='success',1,0))


# In[96]:


data_analysis['normFinal_revenue'].quantile([0.25, 0.5, 0.75])


# In[97]:


data_analysis.loc[data_analysis['normFinal_revenue']<=165.33600,'RevenueDistribution']='Low (0-165 AUD)'
data_analysis.loc[(data_analysis['normFinal_revenue']>165.33600)&(data_analysis['normFinal_revenue']<=281.917478),'RevenueDistribution']='Medium (165-282 AUD)'
data_analysis.loc[(data_analysis['normFinal_revenue']>281.917478)&(data_analysis['normFinal_revenue']<=563.000000),'RevenueDistribution']='High (282-563 AUD)'
data_analysis.loc[data_analysis['normFinal_revenue']>563.000000,'RevenueDistribution']='Highest (563-8267 AUD)'


# In[98]:


data_analysis


# In[99]:


data_analysis['project_duration'].quantile([0.25, 0.5, 0.75])


# In[100]:


import datetime as dt


# In[101]:


data_analysis.loc[data_analysis['project_duration']<=pd.Timedelta(days=73),'ProjectDuration']='< 73 days'
data_analysis.loc[(data_analysis['project_duration']>pd.Timedelta(days=73))&(data_analysis['project_duration']<=pd.Timedelta(days=111)),'ProjectDuration']='73-111 days'
data_analysis.loc[(data_analysis['project_duration']>pd.Timedelta(days=111))&(data_analysis['project_duration']<=pd.Timedelta(days=149)),'ProjectDuration']='111-149 days'
data_analysis.loc[data_analysis['project_duration']>pd.Timedelta(days=149),'ProjectDuration']='> 149 days'


# In[102]:


data_analysis


# In[103]:


#data_analysis[data_analysis['normFinal_revenue']==data_analysis['normFinal_revenue'].max()]


# In[104]:


data_analysis.columns


# In[105]:


data_sankey =data_analysis[['title', 'payment', 'clientType',
       'projectType', 'projectLanguage', 'nameFull', 'country', 'region',
       'final_revenueCurrency', 'successCounter',
       'RevenueDistribution', 'ProjectDuration']]


# In[106]:


data_sankey


# In[107]:


data_sankey.info()


# In[108]:


data_sankey.rename(columns={'title':'Client Name','payment':'Payment','clientType':'Client Type','projectType':'Project Type',                          'projectLanguage':'Project Language','nameFull':'Branch Name','country':'Country','region':'Region',                          'final_revenueCurrency':'Revenue Currency','successCounter':'No. of Successes','RevenueDistribution':'Revenue',                          'ProjectDuration':'Project Duration'},inplace=True)


# In[109]:


data_sankey


# In[110]:


data_sankey.columns


# In[111]:


#sk1 = data_sankey.groupby(['Branch Name','Country'])['Client Name'].count().reset_index()
#sk1.columns = ['source', 'target', 'value']
sk2 = data_sankey.groupby(['Region','Country'])['Client Name'].count().reset_index()
sk2.columns = ['source', 'target', 'value']
sk3 = data_sankey.groupby(['Country','Revenue Currency'])['Client Name'].count().reset_index()
sk3.columns = ['source', 'target', 'value']
sk4 = data_sankey.groupby(['Revenue Currency','Revenue'])['Client Name'].count().reset_index()
sk4.columns = ['source', 'target', 'value']
sk5 = data_sankey.groupby(['Revenue','Project Language'])['Client Name'].count().reset_index()
sk5.columns = ['source', 'target', 'value']
#sk6 = data_sankey.groupby(['Project Duration','Project Type'])['Client Name'].count().reset_index()
#sk6.columns = ['source', 'target', 'value']
sk7 = data_sankey.groupby(['Client Type','Revenue'])['Client Name'].count().reset_index()
sk7.columns = ['source', 'target', 'value']
sk8 = data_sankey.groupby(['Revenue','Project Type'])['Client Name'].count().reset_index()
sk8.columns = ['source', 'target', 'value']
#sk9 = data_sankey.groupby(['Client Type','No. of Successes'])['Client Name'].count().reset_index()
#sk9.columns = ['source', 'target', 'value']
#sk10 = data_sankey.groupby(['No. of Successes','Payment'])['Client Name'].count().reset_index()
#sk10.columns = ['source', 'target', 'value']


# In[112]:


links = pd.concat([sk7,sk8], axis=0)


# In[113]:


links


# In[114]:


unique_source_target = list(pd.unique(links[['source', 'target']].values.ravel('K')))
unique_source_target


# In[115]:


mapping_dict = {k: v for v, k in enumerate(unique_source_target)}
mapping_dict


# In[116]:


links['source'] = links['source'].map(mapping_dict)
links['target'] = links['target'].map(mapping_dict)


# In[117]:


links


# In[118]:


links[links.index==0]


# In[119]:


links2=links.copy()


# In[120]:


import plotly.graph_objects as go


# In[121]:


import random

# Function to generate a random RGBA color with opacity 0.5
def random_rgba_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f'rgba({r}, {g}, {b}, 0.5)'

# Generate a list of 10 random RGBA colors
random_colors = [random_rgba_color() for _ in range(30)]

# Print the list of random colors
random_colors


# In[122]:


colors=pd.DataFrame(random_colors)
colors.rename(columns={0:'colors'},inplace=True)
colors


# In[123]:


links = links.set_index('source')
links=links.join(colors)
links.reset_index(inplace=True)


# In[124]:


links_dict = links.to_dict(orient='list')


# In[125]:


fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = unique_source_target,
        color = random_colors
      
    ),
    link = dict(
      source = links_dict["index"],
      target = links_dict["target"],
      value = links_dict["value"],
      color = links_dict["colors"]
  ))])


# In[126]:


fig.update_layout(title_text="Client-Revenue-Project Sankey Diagram", font_size=10)
fig.update_layout(width=1600, height=1000)
fig.show()


# In[127]:


fig.write_html("C:/Users/KLIN/Downloads/client_revenue_project_sankey_coloured.html")


# In[128]:


green_colors = [
    "rgba(0, 150, 0, 0.5)",
    "rgba(0, 175, 0, 0.5)",
    "rgba(0, 200, 0, 0.5)",
    "rgba(0, 225, 0, 0.5)",
    "rgba(0, 255, 0, 0.5)",
    "rgba(50, 205, 50, 0.5)",
    "rgba(0, 150, 0, 0.5)",
    "rgba(0, 175, 0, 0.5)",
    "rgba(0, 200, 0, 0.5)",
    "rgba(0, 225, 0, 0.5)",
    "rgba(0, 255, 0, 0.5)",
    "rgba(50, 205, 50, 0.5)",
    "rgba(0, 128, 0, 0.5)",
    "rgba(0, 150, 0, 0.5)",
    "rgba(0, 175, 0, 0.5)",
    "rgba(0, 200, 0, 0.5)",
    "rgba(0, 225, 0, 0.5)",
    "rgba(0, 255, 0, 0.5)",
    "rgba(50, 205, 50, 0.5)",
    "rgba(0, 150, 0, 0.5)",
    "rgba(0, 175, 0, 0.5)",
    "rgba(0, 200, 0, 0.5)",
    "rgba(0, 225, 0, 0.5)",
    "rgba(0, 255, 0, 0.5)",
    "rgba(50, 205, 50, 0.5)",
    "rgba(0, 150, 0, 0.5)",
    "rgba(0, 175, 0, 0.5)",
    "rgba(0, 200, 0, 0.5)",
    "rgba(0, 225, 0, 0.5)",
    "rgba(0, 255, 0, 0.5)",
    "rgba(50, 205, 50, 0.5)",
    "rgba(0, 150, 0, 0.5)",
    "rgba(0, 175, 0, 0.5)",
    "rgba(0, 200, 0, 0.5)",
    "rgba(0, 225, 0, 0.5)",
    "rgba(0, 255, 0, 0.5)",
    "rgba(50, 205, 50, 0.5)",
    "rgba(0, 150, 0, 0.5)",
    "rgba(0, 175, 0, 0.5)",
    "rgba(0, 200, 0, 0.5)",
    "rgba(0, 225, 0, 0.5)",
    "rgba(0, 255, 0, 0.5)",
    "rgba(50, 205, 50, 0.5)",
    "rgba(0, 128, 0, 0.5)",
    "rgba(0, 150, 0, 0.5)",
    "rgba(0, 175, 0, 0.5)",
    "rgba(0, 200, 0, 0.5)",
    "rgba(0, 225, 0, 0.5)",
    "rgba(0, 255, 0, 0.5)",
    "rgba(50, 205, 50, 0.5)",
    "rgba(0, 150, 0, 0.5)",
    "rgba(0, 175, 0, 0.5)",
    "rgba(0, 200, 0, 0.5)"
]


# In[129]:


green_colors=pd.DataFrame(green_colors)
green_colors.rename(columns={0:'colors'},inplace=True)
green_colors


# In[130]:


links2


# In[131]:


links2 = links2.set_index('source')
links2=links2.join(green_colors)
links2.reset_index(inplace=True)
links2


# In[132]:


links2_dict = links2.to_dict(orient='list')


# In[133]:


fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = unique_source_target,
      color = 'gray'
    ),
    link = dict(
      source = links2_dict["index"],
      target = links2_dict["target"],
      value = links2_dict["value"],
      color = links2_dict["colors"]
  ))])


# In[134]:


fig.update_layout(title_text="Client-Revenue-Project Sankey Diagram", font_size=10)
fig.update_layout(width=1600, height=1000)
fig.show()


# In[135]:


fig.write_html("C:/Users/KLIN/Downloads/client_revenue_project_sankey.html")


# In[136]:


data_sankey.to_excel("C:/Users/KLIN/Downloads/glt4.xlsx")


# In[ ]:




