# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# INCLUDING THE DATASETS
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# DATA PRE-PROCESSING ON THE COUNTRY NAMES 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

# SPLITTING THE DATASET INTO TRAINING AND TEST DATA
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# TRAINING THE DATASETS
from sklearn.linear_model import LinearRegression
multi_regressor=LinearRegression()
multi_regressor.fit(x_train,y_train)

# PREDICTING THE RESULT
y_pred=multi_regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape( len(y_pred),1 ) ,y_test.reshape( len(y_test),1 )),1))