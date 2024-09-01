#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv("/Users/shuvayandasgupt/Downloads/Heart_Disease_Prediction.csv")


# In[3]:


df.head()


# In[4]:


#info about the dataframe
df.info()


# In[5]:


#fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
#chest pain type(1: typical angina,2: atypical angina,3: non-anginal pain,4: asymptomatic)
#slope: the slope of the peak exercise ST segment(1: upsloping,2: flat,3: downsloping)
#exercise induced angina (1 = yes; 0 = no)
#ekg results: resting electrocardiographic results(0: normal,2: showing probable or definite left ventricular hypertrophy by Estes' criteria)


# In[6]:


df['Heart Disease'].value_counts()


# In[7]:


df.shape


# In[8]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values #target


# In[9]:


df['Sex'] = df['Sex'].astype(str)
df['Heart Disease'] = df['Heart Disease'].astype(str)
sns.countplot(x='Heart Disease', hue = 'Sex', data=df)


# In[10]:


#Male are more prone to heart disease than female


# In[11]:


df['Slope of ST'] = df['Slope of ST'].astype(str)
sns.countplot(x='Heart Disease', hue = 'Slope of ST', data=df)


# In[12]:


#downsloping of "Slope of ST" assures very less chances of heart disease
#Flat of "Sloping of ST" assures higher chances of heart disease


# In[13]:


df['Chest pain type'] = df['Chest pain type'].astype(str)
sns.countplot(x=df['Heart Disease'], hue = 'Chest pain type', data=df)


# In[14]:


#4 represes asymptomic chest pain and it shows that chances of heart disease is higher
#where as non-anginal pain, typical anginal pain is there, very less chance of heart disease


# In[15]:


sns.barplot(x= 'Sex', y = 'BP', data=df)


# In[16]:


#Blood pressure is mostly same in both genders


# In[17]:


sns.barplot(x= 'Sex', y = 'Cholesterol', data=df)


# In[18]:


#Female have higher cholesterol


# In[19]:


sns.barplot(x=df['Heart Disease'], y = df['Cholesterol'], data=df)


# In[20]:


#Although cholesterol is not bigger impact but presence of cholesterol increases the potential of heart disease


# In[21]:


sns.barplot(x=df['Heart Disease'], y = df['BP'], data=df)


# In[22]:


#Similar infrence as cholesterol


# In[23]:


sns.lineplot(x='Age', y = 'BP', data=df)


# In[24]:


#Blood pressure is at its peak at the age of 55-60


# In[25]:


sns.lineplot(x='Age', y = 'Cholesterol', data=df)


# In[26]:


#Blood pressure is at its peak at the age of 50-65


# In[27]:


sns.lineplot(x='BP', y = 'Max HR', data=df)


# In[28]:


#Maximum heartrate is seen at BP b/w 180-200


# In[29]:


sns.barplot(x=df['Heart Disease'], y = df['Max HR'], data=df)


# In[ ]:


sns


# In[30]:


X


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[32]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[34]:


models = {
    'Logistic Regression': LogisticRegression(random_state = 42),
    'Decision Tree': DecisionTreeClassifier(criterion = 'entropy', random_state = 42),
    'Random Forest': RandomForestClassifier(n_estimators = 100,criterion = 'entropy', random_state = 0),
    'SVM': SVC(kernel ='linear', random_state = 0),
    'KNN': KNeighborsClassifier(n_neighbors= 5,p=2, metric='minkowski'),
    'Naive Bayes': GaussianNB() 
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")


# In[35]:


df.columns


# In[ ]:




