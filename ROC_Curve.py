# -*- coding: utf-8 -*-

from pandas.core.dtypes.missing import na_value_for_dtype
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


file = './9to1TrainData.csv'
testFile = './9to1TestData.csv'

df1 = pd.read_csv(testFile, header = 0 ,usecols=[0,2,3,4,5,6,7,8,9], index_col=[0])
df2 = pd.read_csv(testFile, header = 0 ,usecols=[0,10], index_col=[0])

na1 = df1.to_numpy()
na2 = df2.to_numpy()

data = na1
label = na2 


X_train, X_test, y_train, y_test = train_test_split(data,label,test_size = 0.2)

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train,y_train)

y_scores = knn.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()


