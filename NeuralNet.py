#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 18:16:57 2018

@author: codeplayer
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

no_units_hidden_layer = 30
no_inputs = 11
batch_size = 10
number_epochs = 100

classifier = Sequential()
classifier.add(Dense(no_units_hidden_layer,kernel_initializer='uniform',activation='relu',input_dim=no_inputs))
classifier.add(Dense(no_units_hidden_layer,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(no_units_hidden_layer,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(no_units_hidden_layer,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

#Compiling the network
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics= ['accuracy'])

classifier.fit(x=X_train,y=y_train,batch_size = batch_size,epochs=number_epochs)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_test,y_pred)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(no_units_hidden_layer,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(no_units_hidden_layer,kernel_initializer='uniform',activation='relu'))    
    classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifer = KerasClassifier(build_fn=build_classifier,batch_size=batch_size,epochs = number_epochs)
accuracies = cross_val_score( estimator= classifer,X=X_train,y=y_train,cv=10,n_jobs=-1)