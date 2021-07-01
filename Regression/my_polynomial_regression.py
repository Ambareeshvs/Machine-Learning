# IMPORTS
import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd

# INCLUDING THE DATASETS
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

# TRAINING THE DATASET FOR POLYNOMIAL REGRESSION
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=5)
X_poly=poly_regressor.fit_transform(X)
li_regressor_1=LinearRegression()
li_regressor_1.fit(X_poly, y)

# VISUALISING THE POLYNOMIAL REGRESSION
plt.scatter(X, y, color='red')
plt.plot(X,li_regressor_1.predict(X_poly),color='blue')
plt.title('polynomial regression')
plt.xlabel("Position")
plt.ylabel('Salary')
plt.show()

# PREDICTING THE OUTPUT OF POLYNOMIAL REGRESSION
print(li_regressor_1.predict(poly_regressor.fit_transform([[2.5]])))