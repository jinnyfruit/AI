'''
file name: K nearest neighbor Regression algorithm
modified: 2022.06.24
author: jinnyfruit
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

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

# data exploration
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
# plt.show()

# split data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(perch_length,perch_weight, random_state = 42, shuffle = True)

# Change 1 dimensional data into 2 dimensional array
test_array = np.array([1,2,3,4])
print("---------------------- print test array ----------------------")
print(test_array)

test_array = test_array.reshape(2,2)
print("---------------------- print reshaped test array ----------------------")
print(test_array.shape)

X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
print("---------------- print shape of training and test dataset ----------------")
print(X_train.shape ,X_test.shape)

# Import KneighboresRegressor model
knr_model = KNeighborsRegressor()

# Train the model
knr_model.fit(X_train, y_train)

# Test the model
knr_model.score(X_test,y_test)
print("---------------------- Model accuracy score ----------------------")
print(round(knr_model.score(X_test,y_test),2))

# Get the error value between predict value and actual value
test_prediction = knr_model.predict(X_test)

# Use mean absolute error
mae = mean_absolute_error(y_test, test_prediction)
print("---------------------- print mean absolute error ----------------------")
print(round(mae,2))


