#!/usr/bin/env python
# coding: utf-8

# ## (1) 단순 선형회귀

# In[1]:


import numpy as np


# In[2]:


#무작위로 선형 데이터셋 생성
X = 2 * np.random.rand(100,1) #rand: 난수 생성 함수
y = 4 +  3 * X + np.random.randn(100,1) #randn은 정규분포에 대한 난수 생성


# In[3]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[4]:


lin_reg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[5]:


lin_reg.fit(X_train, y_train)


# In[6]:


print(lin_reg.intercept_)  #편향(절편)
print(lin_reg.coef_)       #가중치(기울기)
print(lin_reg.score(X_train, y_train)) #train set 점수
print(lin_reg.score(X_test, y_test)) #test set 점수


# # 1. 다중 Linear Regression
# 
# ## 1.1 데이터 로딩

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[8]:


boston = pd.read_csv('C:/Users/user/Desktop/csv/Boston_house.csv')
boston.head()


# CRIM: 범죄율, INDUS: 비소매상업지역 면적 비율, NOX: 일산화질소 농도, RM: 주택당 방 수
# LSTAT:인구 중 하위 계층 비율, B: 인구 중 흑인 비율, PTRATIO: 학생/교사 비율,
# ZN: 25,000 평방피트 초과 거주지역 비율, CHAS: 찰스강의 경계에 위치한 경우 1, 아니면 0
# AGE: 1940년 이전에 건축된 주택의 비율, RAD: 방사형 고속도로까지의 거리,
# DIS: 직업센터의 거리, TAX: 재산세율

# ## 1.2 모형 만들기

# ### 범죄율, 주택당 방 수, 인구 중 하위 계층 비율, 노후 주택 비율과 주택 가겨의 상관관계 예측하기.

# In[9]:


X = boston[['CRIM','RM', 'LSTAT', 'AGE']]
y = boston[['Target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

get_ipython().run_line_magic('matplotlib', 'inline')

#실제 주택값과 예측한 주택값 간의 상관관계

plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("LINEAR REGRESSION")
plt.show()


# ## 1.3  결과 그리기

# In[10]:


plt.scatter(boston[['CRIM']], boston[['Target']], alpha=0.4)
plt.show()


# In[11]:


plt.scatter(boston[['LSTAT']], boston[['Target']], alpha=0.4)
plt.show()


# In[12]:


plt.scatter(boston[['AGE']], boston[['Target']], alpha=0.4)
plt.show()


# ### 범죄율과 노후 주택 수는 주택 가격과 상관관계가 없어보이고, 방 개수가 가장 큰 양의 상관관계를 보임.
# 
# ### 인구 중 하위 계층 비율은 주택 가격과 음의 상관관계를 보임.

# In[13]:


reg.score(X_train, y_train) #train set R^2 점수


# In[14]:


reg.score(X_test, y_test) #test set R^2 점수


# ## 2. 보스턴 집값 NN

# ### 2-1. keras 데이터 세트 불러오기

# In[15]:


from keras.datasets import boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()


# In[16]:


print(train_data.shape, train_labels.shape)

print(test_data.shape, test_labels.shape)


# In[17]:


import pandas as pd
print(train_labels[:5])

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(train_data, columns=column_names)
df.head()


# ## 2-2. 데이터세트 전처리

# In[18]:


import numpy as np

order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]


# ### 표준화된 데이터 = (원래의 데이터 - 평균) / 표준편차

# In[19]:


mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std

mean = test_data.mean(axis=0)
std = test_data.std(axis=0)
test_data = (test_data - mean) / std


# ## 2-3. 모델링하기

# In[20]:


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))


# In[21]:


from keras.optimizers import Adam

model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae'])


# ## 2-4. 학습

# In[22]:


from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(train_data, train_labels, epochs=500, validation_split=0.2, callbacks=[early_stop])


# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(history.history['mae'], label='train mae')
plt.plot(history.history['val_mae'], label='val mae')
plt.xlabel('epoch')
plt.ylabel('mae [$1,000]')
plt.legend(loc='best')
plt.ylim([0, 5])
plt.show()


# ## 2-5. 테스트용 데이터세트로 학습 확인하기

# In[24]:


test_loss, test_mae = model.evaluate(test_data, test_labels)


# ## 2-6. 예측

# In[25]:


print(np.round(test_labels[:10]))
test_predictions = model.predict(test_data[:10]).flatten()
print(np.round(test_predictions))


# ## 2-7. r2(결정계수) 구하기

# In[26]:


y_train_pred = model.predict(train_data)
y_test_pred = model.predict(test_data)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print("MSE - 트레이닝 데이터: %.2f, 테스트 데이터: %.2f" %(mse_train, mse_test))

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("R2 - 트레이닝 데이터: %.2f, 테스트 데이터: %.2f" %(r2_train, r2_test))


# ## 2-8. 데이터 확인(확인용)

# In[89]:


y_train


# In[27]:


y_train


# In[28]:


y_test.Target


# In[29]:


y_test_pred


# In[30]:


#pred = lr2.predict(X_test)

pred = model.predict(test_data)

aa = np.array(y_test.index.values)

aa


# In[ ]:




