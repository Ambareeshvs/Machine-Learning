# IMPORTING
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING THE DATASETS
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
print(X)
print(y)

# TAKING CARE OF MISSING DATA
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan , strategy="mean")
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
print(X)

# ENCODING THE DATA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[( 'encorder' , OneHotEncoder() , [0])], remainder='passthrough')
X=np.array(ct.fit_transform(X))
print(X)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print(y)

# SPLITTING OF DATASETS
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
print(x_train)
print(y_train)
print(x_test)
print(y_test)

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train[:,3:]=sc.fit_transform(x_train[:,3:])
x_test[:,3:]=sc.transform(x_test[:,3:])
print(x_train)
print(x_test)










