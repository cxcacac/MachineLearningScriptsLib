'''
The codes is copied from:
https://medium.com/datadriveninvestor/linear-regression-using-tensorflow-estimator-9aa570914375#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImQ5NDZiMTM3NzM3Yjk3MzczOGU1Mjg2YzIwOGI2NmU3YTM5ZWU3YzEiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2MDUyMjc0MjEsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjEwODk2NjcxMzIxOTQxMjk5OTU4NSIsImVtYWlsIjoiY3hjYWNhY3llYWhAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImF6cCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsIm5hbWUiOiLpmYjlrZ3ogaoiLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDYuZ29vZ2xldXNlcmNvbnRlbnQuY29tLy0wZGhZQWFUM0VrNC9BQUFBQUFBQUFBSS9BQUFBQUFBQUFBQS9BTVp1dWNrSDNVVnBuYUNmQkdQekM4bXFkVEdMaDFaUi13L3M5Ni1jL3Bob3RvLmpwZyIsImdpdmVuX25hbWUiOiLlrZ3ogaoiLCJmYW1pbHlfbmFtZSI6IumZiCIsImlhdCI6MTYwNTIyNzcyMSwiZXhwIjoxNjA1MjMxMzIxLCJqdGkiOiI3ZjE2YzQyYTBhNWJkOGFlODA4MGE5YTRkZTg4ZjRlN2UzODU4ZmYwIn0.sYn5ujDbVVaYtKXmmsQPSpVeMMvrS9rb32YCd-_795g5kFckp8j_tYXX3C_mPJ5Qc4oWjehVXRdmBQ7BM-wC0fcBcq7JOVyXJEgYQUqd2e-E90hXaAwR-iCpd55XezAD_JmskWlXqNGE8f-UQpabgBgXo2IduqLJGeQ3T-udImmUX_gU-qX5r9zG5Saio-uwsUNBiviLaFCnf-DvM_mFb-nFumq-5ZLUg5nk3vq8Vkm-ax2v9UDpKG05pB5L0yYr8_q_p_22jHBc2Wst39bzBuI5EgT-h9tFMSqtPEwwLsWbc3cYDfF4EijDEao7E7MxOz1ufyeLl3N-wn6Zq2Wgvg
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# get data 

dataset_house = pd.read_csv("d:\\ML-data\\housing.csv")
# dataset_house.head(2) print the first two rows in the dataset.
x_data = dataset_house.drop(['median_house_value'], axis=1)
y = dataset_house['median_house_value']
# find all unique value in ocean_proximity
x_data.ocean_proximity.unique()

# data processing:
# 1. scale the continuous variables
# 2. handle the null values
x_subset= x_data.drop(['ocean_proximity'], axis=1)
x_ocean = x_data['ocean_proximity']
scaler = MinMaxScaler()
x_subset = pd.DataFrame(scaler.fit_transform(x_subset), columns=x_subset.columns, index=x_subset.index)
# concate two dataFrame.
x_data=pd.concat([x_subset, x_ocean], axis=1)
# handle the nulling values with mean values, we can fill it with 0;
x_data['total_bedrooms'].fillna(x_data['total_bedrooms'].mean(), inplace = True)

# creating training and testing data

x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.3)

# converting the raw data to dense tensors
# use tf.feature_column.numeric_column for numeric input that can be directly input to the model
longitude = tf.feature_column.numeric_column('longitude')
latitude = tf.feature_column.numeric_column('latitude')
median_age = tf.feature_column.numeric_column('housing_median_age')
total_rooms = tf.feature_column.numeric_column('total_rooms')
total_bedroom = tf.feature_column.numeric_column('total_bedrooms')
population = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
median_income = tf.feature_column.numeric_column('median_income')
# and we need to convert categorical column ocean_proximity.
# instead of representing the data as a sparse vector
# an embedding column represents lower-dimensional vector containing any number not just 0 or 1;
embedding_size = int(math.floor(len(x_data.ocean_proximity.unique())**0.25))
ocean_proximity=tf.contrib.layers.sparse_column_with_hash_bucket('ocean_proximity',hash_bucket_size=1000)
ocean_proximity=tf.contrib.layers.embedding_column(sparse_id_column=ocean_proximity, dimension=embedding_size)

# establish feature column
feature_col =[latitude, longitude,median_age, total_rooms, total_bedroom , population, households, median_income, ocean_proximity]

# establish the optimizer
opti = tf.train.AdamOptimizer(learning_rate = 0.01)

# define the input function, test function, and eval function
input_func= tf.estimator.inputs.pandas_input_fn(x=x_train, 
                                                y= y_train, 
                                                batch_size=10, 
                                                num_epochs=1000, 
                                                shuffle=True)

test_input_func = tf.estimator.inputs.pandas_input_fn(x= x_test,                                                   
                                                 batch_size=100, 
                                                 num_epochs=1, 
                                                 shuffle=False)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test,
                                                      y=y_test, 
                                                      batch_size=10, 
                                                      num_epochs=1, 
                                                      shuffle=False)
# create a regressor with three hidden layers with units as 9,9,3;
estimator = tf.estimator.DNNRegressor(hidden_units=[9,9,3], feature_columns=feature_col, optimizer=opti, dropout=0.5)

# train the model
estimator.train(input_fn=input_func,steps=20000)
# eval the result
result_eval = estimator.evaluate(input_fn=eval_input_func)

# plot data
predictions=[]
for pred in estimator.predict(input_fn=test_input_func):
    predictions.append(np.array(pred['predictions']).astype(float))
plt.plot(y_test, predictions, 'r*')
plt.xlabel('Actual values')
plt.ylabel('predicted values')

# check the RMSE
np.sqrt(mean_squared_error(y_test, predictions))**0.5