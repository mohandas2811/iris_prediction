#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets,linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[15]:


#data=pd.read_csv()
iris=datasets.load_iris()
iris
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df


# In[19]:


df.isna().sum()


# In[20]:


df.info()


# In[33]:


plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.title("iris")
plt.plot("df")
plt.show()



# In[35]:


sns.jointplot(df)


# In[37]:


sns.pairplot(df)


# In[38]:


sns.boxplot(df)


# In[42]:


sns.lmplot(data=df)


# In[40]:


sns.displot(df)


# In[41]:


sns.violinplot(df)


# In[43]:


sns.scatterplot(df)


# In[44]:


sns.heatmap(df)


# In[46]:


plt.eventplot(df)
plt.show()


# In[47]:


plt.hist(df)
plt.show()


# In[49]:


sns.kdeplot(df)


# In[52]:


sns.histplot(df)


# In[58]:


sns.set_theme(style="darkgrid")
sns.boxplot(df)


# In[79]:


x=df["sepal length (cm)"].values
y=df["sepal width (cm)"].values
from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.50,random_state=0)
x_train



# In[80]:


y_test


# In[81]:


x_train


# In[82]:


y_train


# In[83]:


x_train=x_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)
x_test=x_test.reshape(-1,1)
y_train=y_train.reshape(-1,1)
reg=LinearRegression()
reg.fit(x_train,y_train)


# In[84]:


y_predict=reg.predict(x_test)
y_predict


# In[89]:


x_predict=reg.predict(y_test)
x_predict


# In[90]:


reg.score(x_train,y_train)*100


# In[92]:


reg.score(x_test,y_predict)*100


# In[94]:





# In[ ]:





# In[ ]:




