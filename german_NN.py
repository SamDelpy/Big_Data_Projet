# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:14:29 2019

@author: Ursu
"""

import os
os.chdir("C:/Users/Ursu/Desktop/Bordeaux/Econometrics of big data/ursu/ursu_2019/cours/chapter7/code")
os.getcwd()
bonjour
je teste
un truc

import pandas as pd
import numpy as np
germancredit = pd.read_csv("germancredit.csv",sep=',')
#liste des variables
germancredit.info()
germancredit.Default.value_counts()

un autre essai ici

X = germancredit[['duration','amount','age']]  
y = germancredit.iloc[:, 0]

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#vérification
#Les répartitions des classes sont respectées.
print(np.mean(y_train),np.mean(y_test))

#normalize data
X_train_n = X_train.apply(lambda x:(x-x.min()) / (x.max()-x.min()))
X_test_n = X_test.apply(lambda x:(x-x.min()) / (x.max()-x.min()))
X_train_n.describe()
###############################################################################
############ SingleLayer Perceptron ###########################################
###############################################################################
#keras
from keras.models import Sequential
from keras.layers import Dense
#instanciation du modèle
modelSimple = Sequential()
#architecture
modelSimple.add(Dense(units= 1,input_dim=3,activation="sigmoid"))
#or input_shape=3
# print configuration 
print(modelSimple.get_config())
#compilation algorithme d'apprentissage
modelSimple.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
#apprentissage
modelSimple.fit(X_train_n,y_train,epochs=150,batch_size=10)

#Once the learning is finished, we can display the estimated weights:
print(modelSimple.get_weights())

#Making predictions on new data
predSimple = modelSimple.predict_classes(X_test_n)
print(predSimple[:10])
#confusion matrix
from sklearn import metrics
print( metrics.confusion_matrix(y_test,predSimple))
#succes rate
print(metrics.accuracy_score(y_test,predSimple))

#L’autre solution consiste à utiliser l’outil dédié de la librairie Keras.
score = modelSimple.evaluate(X_test,y_test)
print(score)


###############################################################################
############ Perceptron multicouche ###########################################
###############################################################################
#modélisation
modelMc = Sequential()
modelMc.add(Dense(units=4,input_dim=3,activation="relu"))
modelMc.add(Dense(units=4,activation="sigmoid"))
#modelMc.add(Dense(units=300,activation="sigmoid"))
modelMc.add(Dense(units=1,activation="sigmoid"))

#compiling the model
modelMc.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
#apprentissage
modelMc.fit(X_train,y_train,validation_split=0.2,epochs=150,batch_size=10)
#poids synaptiques
print(modelMc.get_weights())

#score
score = modelMc.evaluate(X_test,y_test)
print(score)

predMc = modelMc.predict_classes(X_test)
from sklearn import metrics
print( metrics.confusion_matrix(y_test,predMc))

#taux de succès
print(metrics.accuracy_score(y_test,predMc))


###############################################################################
############ MultiLayer Perceptron  ###########################################
###############################################################################
import os
os.chdir("C:/Users/Ursu/Desktop/Bordeaux/Econometrics of big data/ursu/ursu_2019/cours/chapter7/code")
os.getcwd()

import pandas as pd
import numpy as np

X = pd.read_csv("germancredit.csv",sep=',')
#liste des variables
X.info()
X.Default.value_counts()
X = pd.get_dummies(X)
y = X['Default']

# Remove the labels from the features
# axis 1 refers to the columns
X = X.drop('Default', axis = 1)

# Saving feature names for later use
feature_list = list(X.columns)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 42)

#vérification
#Les répartitions des classes sont respectées.
print(np.mean(y_train),np.mean(y_test))

#normalize data
X_train_n = X_train.apply(lambda x:(x-x.min()) / (x.max()-x.min()))
X_test_n = X_test.apply(lambda x:(x-x.min()) / (x.max()-x.min()))
X_train_n.describe()

#keras
from keras.models import Sequential
from keras.layers import Dense

modelMc = Sequential()
modelMc.add(Dense(units=10,input_dim=61,activation="relu"))
modelMc.add(Dense(units=20,activation="sigmoid"))
modelMc.add(Dense(units=30,activation="sigmoid"))
#modelMc.add(Dense(units=300,activation="sigmoid"))
modelMc.add(Dense(units=1,activation="sigmoid"))

#compilation algorithme d'apprentissage
modelMc.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
#apprentissage
modelMc.fit(X_train,y_train,epochs=100,batch_size=5)

#score
score = modelMc.evaluate(X_test,y_test)
print(score)

predMc = modelMc.predict_classes(X_test)
from sklearn import metrics
print( metrics.confusion_matrix(y_test,predMc))

#taux de succès
print(metrics.accuracy_score(y_test,predMc))

