# Financial Technology - Final Project kNN 9 t0 1 ML Part

## Needed Python Extension Packages (Python 3.9.7)

must import **numpy , pandas , sklearn** Packages in order to run the program

```python

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

```
for **kNN_sklearn.py**

```python

from pandas.core.dtypes.missing import na_value_for_dtype
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

```
for **kNN_GS.py** 

## Program Description

All the python files is included in the **"9to1_kNN.ipynb"** and the operation result is in the **"result"** folder.

The main file **"kNN_sklearn.py"** use dataset **"9to1TestData.csv"** to train and test the kNN model and the **"kNN_GS.py"**  use gridsearchcv to tune the model.\
Run **"kNN_GS.py"** first and get the **best parameter = { n_neighbors = 9, leaf_size = 20, p = 1, weights = 'uniform', metric = 'chebyshev'}**\
Then use this parameter to predict and output the result in **"9to1_kNN.csv"** 

**PS:** the source python files is in the **"py_files"** folder , the best accuracy and precision score tested is in **"9to1_kNN_precision.txt"** is included in the **"result"** folder , and **"ROC_Curve.py"** is for plot ROC Curve


## Reference

**機器學習：KNN分類演算法 - iT 邦幫忙**- https://ithelp.ithome.com.tw/articles/10197110 \
**KNN Classification using Sklearn Python** - DataCamp- https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn \
**How to tune the K-Nearest Neighbors classifier with Scikit-Learn in Python — DataSklr** - https://www.datasklr.com/select-classification-methods/k-nearest-neighbors
