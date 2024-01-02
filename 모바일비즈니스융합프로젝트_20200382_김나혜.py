#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
df = pd.read_csv('jejuCard.csv', encoding='cp949') #/Users/kimnahye/Desktop/jejuCard.csv


# In[29]:


df


# In[4]:


df.info()


# In[5]:


df.columns = df.columns.str.replace(' ', '')


# In[6]:


df


# In[7]:


df.loc[df['카드이용금액']==df['카드이용금액'].max(), ['업종명','성별','연령대별','카드이용금액']]


# In[8]:


df.loc[df['카드이용금액']==df['카드이용금액'].min(), ['업종명','성별','연령대별','카드이용금액']]


# In[9]:


df.loc[df['카드이용건수']==df['카드이용건수'].max(), ['업종명','성별','연령대별','카드이용건수']]


# In[10]:


df.loc[df['카드이용건수']==df['카드이용건수'].min(), ['업종명','성별','연령대별','카드이용건수']]


# In[11]:


df.loc[df['건당이용금액']==df['건당이용금액'].max(), ['업종명','성별','연령대별','건당이용금액']]


# In[12]:


df.loc[df['건당이용금액']==df['건당이용금액'].min(), ['업종명','성별','연령대별','건당이용금액']]


# In[13]:


age = df[df['연령대별']== '20대']
df_20 = pd.concat([df, age], ignore_index=True)
df_20 = df_20.iloc[13146:]
df_20


# In[14]:


age = df[df['연령대별']== '30대']
df_30 = pd.concat([df, age], ignore_index=True)
df_30 = df_30.iloc[13146:]
df_30


# In[15]:


age = df[df['연령대별']== '40대']
df_40 = pd.concat([df, age], ignore_index=True)
df_40 = df_40.iloc[13146:]
df_40


# In[16]:


age = df[df['연령대별']== '50대']
df_50 = pd.concat([df, age], ignore_index=True)
df_50 = df_50.iloc[13146:]
df_50


# In[27]:


plt.barh(df['업종명'], df['카드이용금액'], color = 'skyblue')
plt.title('소비가 가장 높은 업종은?', size=20)
plt.xlabel('카드이용금액')
plt.ylabel('업종명')
plt.tick_params(labelsize=9)
# plt.rc('font', family='AppleGothic') 맥북인 경우에 사용
plt.rc('font', family='Malgun Gothic')
plt.show()


# In[18]:


x=df['제주중분류']
y=df['카드이용금액']
plt.bar(x, y, color = 'skyblue')
plt.title('제주 지역별 카드이용금액 분포',size=20)
plt.xlabel('제주 지역')
plt.ylabel('카드이용금액')
# plt.rc('font', family='AppleGothic')
plt.rc('font', family='Malgun Gothic')
plt.tick_params(labelsize=9)
plt.show()


# In[19]:


plt.barh(df_20['업종명'], df_20['카드이용금액'], color='skyblue')
plt.title('20대의 소비가 가장 높은 업종은?', size=20)
plt.xlabel('카드이용금액')
plt.ylabel('업종명')
plt.tick_params(labelsize=9)
# plt.rc('font', family='AppleGothic')
plt.rc('font', family='Malgun Gothic')
plt.show()


# In[20]:


plt.barh(df_30['업종명'], df_30['카드이용금액'], color='skyblue')
plt.title('30대의 소비가 가장 높은 업종은?', size=20)
plt.xlabel('카드이용금액')
plt.ylabel('업종명')
plt.tick_params(labelsize=9)
# plt.rc('font', family='AppleGothic')
plt.rc('font', family='Malgun Gothic')
plt.show()


# In[21]:


plt.barh(df_40['업종명'], df_40['카드이용금액'], color='skyblue')
plt.title('40대의 소비가 가장 높은 업종은?', size=20)
plt.xlabel('카드이용금액')
plt.ylabel('업종명')
plt.tick_params(labelsize=9)
# plt.rc('font', family='AppleGothic')
plt.rc('font', family='Malgun Gothic')
plt.show()


# In[22]:


plt.barh(df_50['업종명'], df_50['카드이용금액'], color='skyblue')
plt.title('50대의 소비가 가장 높은 업종은?', size=20)
plt.xlabel('카드이용금액')
plt.ylabel('업종명')
plt.tick_params(labelsize=9)
# plt.rc('font', family='AppleGothic')
plt.rc('font', family='Malgun Gothic')
plt.show()


# In[23]:


plt.barh(df_20['제주중분류'], df_20['카드이용금액'], color='skyblue')
plt.title('20대의 소비가 가장 높은 지역은?', size=20)
plt.xlabel('카드이용금액')
plt.ylabel('제주 지역')
plt.tick_params(labelsize=9)
# plt.rc('font', family='AppleGothic')
plt.rc('font', family='Malgun Gothic')
plt.show()


# In[24]:


plt.barh(df_30['제주중분류'], df_30['카드이용금액'], color='skyblue')
plt.title('30대의 소비가 가장 높은 지역은?', size=20)
plt.xlabel('카드이용금액')
plt.ylabel('제주 지역')
plt.tick_params(labelsize=9)
# plt.rc('font', family='AppleGothic')
plt.rc('font', family='Malgun Gothic')
plt.show()


# In[25]:


plt.barh(df_40['제주중분류'], df_40['카드이용금액'], color='skyblue')
plt.title('40대의 소비가 가장 높은 지역은?', size=20)
plt.xlabel('카드이용금액')
plt.ylabel('제주 지역')
plt.tick_params(labelsize=9)
# plt.rc('font', family='AppleGothic')
plt.rc('font', family='Malgun Gothic')
plt.show()


# In[26]:


plt.barh(df_50['제주중분류'], df_50['카드이용금액'], color='skyblue')
plt.title('50대의 소비가 가장 높은 지역은?', size=20)
plt.xlabel('카드이용금액')
plt.ylabel('제주 지역')
plt.tick_params(labelsize=9)
# plt.rc('font', family='AppleGothic')
plt.rc('font', family='Malgun Gothic')
plt.show()


# In[ ]:




