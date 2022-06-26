#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("fish.csv")


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.isnull().sum()


# In[9]:


data.duplicated()


# In[10]:


data.shape


# In[11]:


data.columns


# In[12]:


data.nunique()


# In[13]:


plt.plot(data['Species'])
plt.show()


# In[14]:


plt.hist(data['Species'], bins = 10)
plt.show()


# In[15]:


data['Species'].value_counts()


# In[16]:


plt.figure(figsize = (15,6))
sns.countplot('Species',data = data,palette = 'hls')
plt.show()


# In[17]:


fish_weight = data['Weight']
q3 = fish_weight.quantile(0.75)
q1 = fish_weight.quantile(0.25)
IQR = q3 - q1
lower_limit = q1 - (1.5 * IQR)
upper_limit = q3 + (1.5 * IQR)


# In[18]:


weight_outliers = fish_weight[(fish_weight < lower_limit) | (fish_weight > upper_limit)]
weight_outliers


# In[19]:


fish_Length1 = data['Length1']
q3 = fish_Length1.quantile(0.75)
q1 = fish_Length1.quantile(0.25)
IQR = q3 - q1
lower_limit = q1 - (1.5 * IQR)
upper_limit = q3 + (1.5 * IQR)
Length1_outliers = fish_Length1[(fish_Length1 < lower_limit) | (fish_Length1 > upper_limit)]
Length1_outliers


# In[20]:


fish_Length2 = data['Length2']
q3 = fish_Length2.quantile(0.75)
q1 = fish_Length2.quantile(0.25)
IQR = q3 - q1
lower_limit = q1 - (1.5 * IQR)
upper_limit = q3 + (1.5 * IQR)
Length2_outliers = fish_Length2[(fish_Length2 < lower_limit) | (fish_Length2 > upper_limit)]
Length2_outliers


# In[21]:


fish_Length3 = data['Length3']
q3 = fish_Length3.quantile(0.75)
q1 = fish_Length3.quantile(0.25)
IQR = q3 - q1
lower_limit = q1 - (1.5 * IQR)
upper_limit = q3 + (1.5 * IQR)
Length3_outliers = fish_Length3[(fish_Length3 < lower_limit) | (fish_Length3 > upper_limit)]
Length3_outliers


# In[22]:


plt.figure(figsize = (15,6))
sns.boxplot(data['Height'])
plt.xticks(rotation = 90)
plt.show()


# In[23]:


data[142:145]


# In[24]:


data_new = data.drop([142,143,144])


# In[25]:


data_new.head()


# In[26]:


from sklearn.preprocessing import StandardScaler


# In[27]:


scaler = StandardScaler()


# In[28]:


s_columns = ['Weight','Length1','Length2','Length3','Height','Width']
data_new[s_columns] = scaler.fit_transform(data_new[s_columns])
data_new.describe()


# In[29]:


from sklearn.preprocessing import LabelEncoder


# In[30]:


le = LabelEncoder()
data_new['Species'] = le.fit_transform(data_new['Species'].values)


# In[31]:


data_clean = data_new.drop("Weight", axis = 1)
y = data_new['Weight']


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


x_train, x_test, y_train, y_test = train_test_split(data_clean,y,test_size = 0.2,random_state = 42)


# In[34]:


from sklearn.ensemble import RandomForestRegressor


# In[35]:


model = RandomForestRegressor()
model.fit(x_train,y_train)


# In[36]:


y_pred = model.predict(x_test)


# In[37]:


y_pred


# In[38]:


print('Traning accuracy :',model.score(x_train,y_train))
print('Test accuracy :',model.score(x_test,y_test))


# In[39]:


from sklearn.linear_model import LinearRegression


# In[40]:


lrmodel =LinearRegression()


# In[41]:


lrmodel.fit(x_train,y_train)


# In[42]:


lr_pred = lrmodel.predict(x_test)


# In[43]:


y_pred


# In[44]:


print('Traning accuracy :',lrmodel.score(x_train,y_train))
print('Test accuracy :',lrmodel.score(x_test,y_test))


# In[45]:





# In[47]:


import xgboost as xgb
xgb1 = xgb.XGBRegressor()


# In[48]:


xgb1.fit(x_train,y_train)
xgb_pred = xgb1.predict(x_test)


# In[49]:


xgb1.save_model("model.json")


# In[50]:





# In[51]:


import streamlit as  st


# In[53]:


st.header("Fish Weight Prediction App")
st.text_input("Enter your name :",key = "name")


# In[55]:


np.save('classes.npy',le.classes_)


# In[56]:


le.classes_ = np.load('classes.npy',allow_pickle = True)


# In[57]:


xgb_best = xgb.XGBRegressor()


# In[60]:


xgb_best.load_model("model.json")


# In[61]:


if st.checkbox('Show Training Dataframe'):
    data


# In[62]:


st.subheader("please select relevant features of your fish")
left_column, right_column = st.columns(2)
with left_column:
    inp_species = st.radio('Name of the fish : ',np.unique(data['Species']))


# In[64]:


input_Length1 = st.slider('Vertical length(cm)',0.0,max(data["Length1"]))
input_Length2 = st.slider('Diagonal length(cm)',0.0,max(data["Length2"]))
input_Length3 = st.slider('Cross length(cm)',0.0,max(data["Length3"]))
input_Height = st.slider('Height (cm)',0.0,max(data["Height"]))
input_Width = st.slider('Width(cm)',0.0,max(data["Width"]))


# In[65]:


if st.button('Make Prediction'):
    input_species = le.transform(np.expand_dims(inp_species,-1))
    inputs = np.expand_dims(
                            [int(input_species),input_Length1,input_Length2,input_Length3,input_Height,input_Width],axis = 1)
    prediction = xgb_best.predict(inputs)
    print("final pred :",np.squeeze(prediction,-1))
    st.write(f"your fish weight is: {np.squeeze(prediction, -1):.2f}g")


# In[ ]:




