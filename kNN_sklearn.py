# -*- coding: utf-8 -*-

from pandas.core.dtypes.missing import na_value_for_dtype
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd


# Load files get datas

file = './9to1TrainData.csv'
testFile = './9to1TestData.csv'

df1 = pd.read_csv(testFile, header = 0 ,usecols=[0,2,3,4,5,6,7,8,9], index_col=[0])
df2 = pd.read_csv(testFile, header = 0 ,usecols=[0,10], index_col=[0])

na1 = df1.to_numpy()
na2 = df2.to_numpy()

data = na1
label = na2 

# Spitting datas

train_data , test_data , train_label , test_label = train_test_split(data,label,test_size = 0.2)

# Use KNeighborsClassifier and train the model (Parameter from GS)

knn = KNeighborsClassifier(n_neighbors = 9, leaf_size = 20, p = 1, weights = 'uniform', metric = 'chebyshev' , algorithm='brute') # 試驗後發現 algorithm='brute' 的效果最好
knn.fit(train_data,train_label)


df3 = knn.predict(test_data)
df4 = test_label

y_pred = df3
y_true = df4

# Get accuracy score and precision score

a = accuracy_score(y_true, y_pred)
print(a)
p = precision_score(y_true, y_pred, average=None)
print(p)

testDF = pd.read_csv(testFile, header = 0 ,usecols=[0,2,3,4,5,6,7,8,9], index_col=[0])
testNP = testDF.to_numpy()
test_pred = knn.predict(testNP)
index = pd.Series(data = testDF.index)
test_pred = pd.Series(data=test_pred)
output = pd.concat({'代號': index, 'label': test_pred},
                        axis=1)


# Outout the data
output.to_csv('./9to1_kNN.csv',index = False)



