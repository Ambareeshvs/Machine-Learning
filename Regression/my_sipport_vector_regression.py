# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# INCLUDING THE DATASETS
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
y=y.reshape(len(y),1)

# FEATURE SCALING THE DATASETS
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)

# TRAINING THE DATASETS
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

# PREDICTING THE OUTPUT
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))

# VISUALISING THE TRAINING SET
x_grid=np.arange(min(sc_x.inverse_transform(X)),max(sc_x.inverse_transform(X)),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(y),color='red')
plt.plot(x_grid,sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))),color='blue')
plt.title('Supprot Vector Regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()





