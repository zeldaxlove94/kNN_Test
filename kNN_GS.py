# -*- coding: utf-8 -*-

from pandas.core.dtypes.missing import na_value_for_dtype
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

#Parameter tuning with GridSearchCV 

estimator_KNN = KNeighborsClassifier(algorithm='auto')
parameters_KNN = {'n_neighbors': (9,10,11),
                  'leaf_size': (20,40),
                  'p': (1,2),
                  'weights': ('uniform', 'distance'),
                  'metric': ('minkowski', 'chebyshev'),
}
                   
# with GridSearch

grid_search_KNN = GridSearchCV(
    estimator=estimator_KNN,
    param_grid=parameters_KNN,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 3
)

KNN_1=grid_search_KNN.fit(train_data, train_label)
y_pred_KNN1 =KNN_1.predict(test_data)

#Parameter setting that gave the best results on the hold out data.

print(grid_search_KNN.best_params_ ) 

#Mean cross-validated score of the best_estimator

print('Best Score - KNN:', grid_search_KNN.best_score_ )

{'leaf_size': 20, 'metric': 'minkowski', 'n_neighbors': 10, 'p': 1, 'weights': 'uniform'}


