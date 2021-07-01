# IMPORTS
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# INCLUDING THE DATASETS
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# SPLITING THE DATA
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0) 

# TRAINING THE TRAIN DATASET
from sklearn.linear_model import LinearRegression
l_regressor = LinearRegression()
l_regressor.fit(x_train,y_train)

# PREDICTING THE VALUE
y_pred=l_regressor.predict(x_test)

# PLOTTING THE PREDICTIONS OF THE TRAINING DATASET
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,l_regressor.predict(x_train),color='blue')
plt.xlabel('years of experience')
plt.ylabel('salary(training)')
plt.title('experience vs salary')
plt.show()

# PLOTTING THE PREDICTIONS OF THE TESTING DATASETS
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,l_regressor.predict(x_test),color='blue')
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary(testing')
plt.show()


