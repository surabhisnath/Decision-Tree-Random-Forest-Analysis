
# coding: utf-8

# In[4]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# In[8]:


clfdt = pickle.load(open('clfdt.pkl', 'rb'))
clfrf = pickle.load(open('clfrf.pkl', 'rb'))


# In[9]:


train_data = pd.read_csv('./Training.csv')
train_data = np.array(train_data)
train_labels = train_data[:,1]
train_features = train_data[:,2:]

test_data = pd.read_csv('./Test.csv')
test_data = np.array(test_data)
test_labels = test_data[:,1]
test_features = test_data[:,2:]


# In[10]:


pred_dt = clfdt.predict(test_features)
print(accuracy_score(pred_dt, test_labels))

pred_rf = clfrf.predict(test_features)
print(accuracy_score(pred_rf, test_labels))

