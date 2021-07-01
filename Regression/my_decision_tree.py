# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# INCLUDING THE DATASETS
dataset=pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,:-1].values

# TRAINING THE DATASETS
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#PREDICTING THE OUTPUT
regressor.predict([[6.5]]) 

# VISUALISING THE RESULT SINCE POSSIBLE BCOZ ONLY ONE FEATURE
x_grid=np.arange(min(X),max(X),0.1)
x_grid=x_grid.reshape((len(x_grid)),1)
plt.scatter(X,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Decision Tree')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()