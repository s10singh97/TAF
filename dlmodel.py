import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Datasets/indiadata.csv", encoding="latin-1")
X = dataset.iloc[:, 1:2].values
y1 = dataset.iloc[:, 4:5].values
y2 = dataset.iloc[:, 5:6].values

from sklearn.preprocessing import MinMaxScaler
sc1 = MinMaxScaler(feature_range=(0,1))
X_train1 = sc1.fit_transform(X)
y_train1 = sc1.transform(y1)
sc2 = MinMaxScaler(feature_range=(0,1))
X_train2 = sc2.fit_transform(X)
y_train2 = sc2.transform(y2)

# Reshaping
X_train1 = np.reshape(X_train1, (X_train1.shape[0], X_train1.shape[1], 1))
X_train2 = np.reshape(X_train2, (X_train2.shape[0], X_train2.shape[1], 1))

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

def lat():
    regressor = Sequential()
    # First LSTM Layer
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train1.shape[1], 1)))
    regressor.add(Dropout(0.2))
    # Second LSTM Layer
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Third LSTM Layer
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Fourth LSTM Layer
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    # Output Layer
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # Fitting the RNN to the training set
    regressor.fit(X_train1, y_train1, batch_size = 32, epochs = 100)
    
    input_val = sc1.transform(2018)
    input_val = np.reshape(input_val, (input_val.shape[0], input_val.shape[1], 1))
    y_pred1 = regressor.predict(input_val)
    y_pred1 = sc1.inverse_transform(y_pred1)
    return y_pred1

def lng():
    regressor = Sequential()
    # First LSTM Layer
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train2.shape[1], 1)))
    regressor.add(Dropout(0.2))
    # Second LSTM Layer
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Third LSTM Layer
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Fourth LSTM Layer
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    # Output Layer
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # Fitting the RNN to the training set
    regressor.fit(X_train2, y_train2, batch_size = 32, epochs = 100)
    
    input_val = sc2.transform(2018)
    input_val = np.reshape(input_val, (input_val.shape[0], input_val.shape[1], 1))
    y_pred2 = regressor.predict(input_val)
    y_pred2 = sc2.inverse_transform(y_pred2)
    return y_pred2

latitude = lat()
latitude = latitude.tolist()
for i in latitude:
    latitude = i
for i in latitude:
    latitude = float("{0:.6f}".format(i))
longitude = lng()
longitude = longitude.tolist()
for i in longitude:
    longitude = i
for i in longitude:
    longitude = float("{0:.6f}".format(i))

from urllib.request import urlopen
import json

def getplace(lat, lon):
    url = "http://maps.googleapis.com/maps/api/geocode/json?"
    url += "latlng=%s,%s&sensor=false" % (lat, lon)
    while True:
        try:
            v = urlopen(url).read()
            j = json.loads(v)
            components = j['results'][0]['formatted_address']
            flag = 0        # Successful data transfer in components 
            print(components)
        except:
            flag = 1        # Unsuccessful data transfer in components
        if flag == 0:
            break
        
getplace(latitude, longitude)