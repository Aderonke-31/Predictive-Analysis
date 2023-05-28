#!/usr/bin/env python
# coding: utf-8

# In[43]:


# Installing dependencies
import pandas as pd
import seaborn as sns
import numpy as np


# In[2]:


data = pd.read_csv("C:/Users/ADERONKE/Downloads/archive/House_Data.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.columns


# In[6]:


data.shape


# In[7]:


data.describe()


# In[9]:


# Data Visualisation for checking the relationship between variables


# In[10]:


# Data visualisation is a process which you can visualise your data and the relationships btw variables


# In[11]:


data.isnull().sum()


# In[13]:


data_drop_column = data.dropna(axis=1)


# In[14]:


data_drop_column.shape


# In[18]:


data.dtypes


# In[20]:


data.location.dtypes


# In[25]:


# Visualisation


# In[26]:


# Checking r/ship between variables


# In[27]:


data.head()


# In[62]:


sns.relplot(x='price', y='size' ,data=data)


# In[63]:


sns.relplot(x='price', y='location' ,data=data)


# In[46]:


sns.relplot(x='price', y='area_type', hue='balcony', data=data)


# In[48]:


# Model


# In[49]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[50]:


train = data.drop(['price', 'bath', 'availability'], axis=1)
test = data['price']


# In[57]:


x_train, x_test, y_train, y_test = train_test_split(train, test, test_size=0.3, random_state=2)


# In[55]:


regr = LinearRegression()


# In[ ]:




