#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv(r"D:\Iris.csv")


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# In[10]:


df.info()


# In[11]:


df.corr()


# In[12]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,cmap='tab20',vmin=0,vmax=1)


# In[13]:


import warnings 
warnings.filterwarnings('ignore')


# In[14]:


a = df.hist(figsize=(10,10) , bins = 10)


# In[15]:


import seaborn as sns
a = df.columns
for i in a:
    sns.distplot(a = df[i])
    plt.show()


# In[16]:


X= df[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values


# In[17]:


y=df['Species']


# In[18]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[21]:


test = SelectKBest(score_func=chi2, k=5)
fit = test.fit(X, y)

# Summarize scores
np.set_printoptions(precision=5)
print(fit.scores_)

features = fit.transform(X)
# Summarize selected features
print(features[0:10,:])


# In[22]:


from sklearn import preprocessing

X = np.asarray(X)
  
X = preprocessing.StandardScaler().fit(X).transform(X)


# In[24]:


X


# In[40]:


y=df['Species']


# In[41]:


y


# In[42]:


y.unique()


# In[43]:


y.value_counts()


# In[44]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)


# In[45]:


X_train.shape


# In[46]:


X_test.shape


# In[47]:


X_train


# In[48]:


y_train.shape


# In[49]:


y_test


# In[57]:


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,classification_report , confusion_matrix


# KNN algorithm

# In[58]:


def knn_classifier(X_train, X_test, y_train, y_test):
    classifier_knn = KNeighborsClassifier(metric = 'minkowski', p = 5)
    classifier_knn.fit(X_train, y_train)

    y_pred = classifier_knn.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    
    return print(f"Train score : {classifier_knn.score(X_train, y_train)}\nTest score : {classifier_knn.score(X_test, y_test)}\nAccuracy score:{accuracy_score(y_test,y_pred)}\nCR:{classification_report(y_test,y_pred)}")


# Decision Tree

# In[59]:


def tree_classifier(X_train, X_test, y_train, y_test):
    
    classifier_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier_tree.fit(X_train, y_train)

    y_pred = classifier_tree.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return print(f"Train score : {classifier_tree.score(X_train, y_train)}\nTest score : {classifier_tree.score(X_test, y_test)}\nAccuracy score :{accuracy_score(y_test, y_pred)}\nCR:{classification_report(y_test,y_pred)}")


# In[60]:


def print_score(X_train, X_test, y_train, y_test):
    print("KNN:\n")
    knn_classifier(X_train, X_test, y_train, y_test)
    
    print("-"*100)
    print()
    
    print("Decision Tree:\n")
    tree_classifier(X_train, X_test, y_train, y_test)


# In[61]:


print_score(X_train, X_test, y_train, y_test)

