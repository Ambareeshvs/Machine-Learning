# IMPORTS
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# INCLUDING THE DATASETS
datasets = pd.read_csv('Position_Salaries.csv')
X = datasets.iloc[:,1:-1].values
y = datasets.iloc[:,-1].values

# TRAING THE DATASETS
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,y)

# PREDICTING THE OUTPUT
regressor.predict([[6.5]])

# VISUALISING THE PREDICTIONS
x_grid = np.arange(min(X),max(X),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(X,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title("Random Forest Regression")
plt.xlabel('positions')
plt.ylabel('salary')
plt.show()
