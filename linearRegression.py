'''
file name: Linear Regression
modified: 2022.06.27
author: jinnyfruit
'''
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read the perch data
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# Split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(perch_length, perch_weight, random_state=42, shuffle=True)
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)

# Linear Regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Predict 50cm perch weight
print("\n---------------------- prediction of 50cm perch ----------------------")
predict = float(lr_model.predict([[50]]))
print(round(predict,2))

# get weight and bias of linear equation
# coef_ and intercept_ is model parameter
print("\n---------------------- print weight and bias of Linear model ----------------------")
print(lr_model.coef_, lr_model.intercept_)

# Draw training set scatterplot
plt.scatter(X_train, y_train)

# Draw a one-dimensional equation
plt.plot([15,50], [15*lr_model.coef_ + lr_model.intercept_, 50*lr_model.coef_ + lr_model.intercept_])

# data of 50cm perch
plt.scatter(50, 1241.8, marker = '^')
plt.xlabel('length')
plt.ylabel('length')

# print score of Linear Regression model
print("\n---------------------- Score of linear Regression model ( training set )  ----------------------")
print(round(float(lr_model.score(X_train,y_train)),2))
print("\n---------------------- Score of linear Regression model ( test set )  ----------------------")
print(round(float(lr_model.score(X_test,y_test)),2))

plt.show()