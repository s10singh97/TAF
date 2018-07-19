import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Datasets/indiadata.csv", encoding="latin-1")
X = dataset.iloc[:, 1:2].values
y1 = dataset.iloc[:, 4:5].values
y2 = dataset.iloc[:, 5:6].values

from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size = 0.2, random_state = 0)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train1 = sc.fit_transform(X_train1)
X_test1 = sc.transform(X_test1)
X_train2 = sc.fit_transform(X_train2)
X_test2 = sc.transform(X_test2)

import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train1, y_train1, batch_size = 10, epochs = 100)
y1_pred = classifier.predict(2018)
classifier.fit(X_train2, y_train2, batch_size = 10, epochs = 100)
y2_pred = classifier.predict(2018)