'''
file name: Multiple Regression
modified: 2022.07.05
author: jinnyfruit
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Read data
df = pd.read_csv('data/perch_full.csv')
print("\n---------------------- print perch data ----------------------")
print(df.describe())

perch_data = df.to_numpy()
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# Split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(perch_data, perch_weight, random_state=42, shuffle=True)

# Learn about the fit method
# fit() =  Increases the number of elements by multiplying each element value
poly = PolynomialFeatures()
print("\n---------------------- before fit() ----------------------")
print([[2,3]])
poly.fit([[2,3]])
print("\n---------------------- after fit() ----------------------")
print(poly.transform([[2,3]]))     # 1,2,3,4,6,9 -> 1,2,3, 2*2, 2*3, 3*3

# Transform data
poly = PolynomialFeatures(include_bias=False)    # Get rid of bias
poly.fit(X_train)
train_poly = poly.transform(X_train)
test_poly = poly.transform(X_test)
print("\n---------------------- print shape of train data ----------------------")
print(train_poly.shape)

# get_feature_names() is going to disappear as the version is upgraded,
# so I used get_feature_names_out(), which is compatible with the next version.
print("\n---------------------- print feature name ----------------------")
print(poly.get_feature_names_out())

# Get Model
multiple_model = LinearRegression()
multiple_model.fit(train_poly, y_train)

# Get the score of model
print("\n---------------------- Training data Model score ----------------------")
print(round(float(multiple_model.score(train_poly,y_train)),2))
print("\n---------------------- Test data Model score ----------------------")
print(round(float(multiple_model.score(test_poly,y_test)),2))

# What if we get more feature
# We can specify the maximum degree of the higher order using "degree"
poly = PolynomialFeatures(degree = 5, include_bias = False)
poly.fit(X_train)
train_poly = poly.transform(X_train)
test_poly = poly.transform(X_test)
print("\n---------------------- get the train poly shape (degree = 5) ----------------------")
print(train_poly.shape)     # feature increases upto 55

multiple_model.fit(train_poly, y_train)
print("\n---------------------- Training data Model score (2) ----------------------")
print(round(float(multiple_model.score(train_poly,y_train)),2))
print("\n---------------------- Test data Model score (2) ----------------------")
print(round(float(multiple_model.score(test_poly,y_test)),2))
print("\n---------------------- Overfitting occured ----------------------")

# Regularization using Scaler
scaler = StandardScaler()
scaler.fit(train_poly)
train_scaled = scaler.transform(train_poly)
test_scaled = scaler.transform(test_poly)


