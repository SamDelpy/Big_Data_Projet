# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:24:08 2020

@author: andria
"""

import os
os.chdir("C:/Users/andri/Desktop/Econometrics of Big Data/homework3")
os.getcwd()

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns

from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
"""
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import acf, pacf, graphics
from statsmodels.tsa.stattools import adfuller
"""

data = pd.read_csv("AR.txt", header = None)
plt.plot(data.index+1,data) #semble fluctuer autour de 0, pas de tendance particulière
#`la serie est donc stationnaire

lag_plot(data) # a priori pas de corrélation

df_corr = pd.concat([data.shift(1),data],axis=1)
df_corr.columns=['t-1','t+1']
df_corr.corr(method="pearson")
sns.heatmap(df_corr.corr(method="pearson"),cmap="Blues",annot=True)
# montre qu'il n'y a aucune corrélation entre une période t et t-1

autocorrelation_plot(data)
plot_acf(data, lags = 51) #meme graphique que précédent mais plus lisible

X = data.values

train,test = X[0:len(X)-50],X[len(X)-50:]

model = AR(train)
model_fit= model.fit()

window = model_fit.k_ar #Variables
coeff = model_fit.params # Coefficients

history = train[len(train) - window:]
history = [history[i] for i in range(len(history))]
history

predictions=[]

for t in test:
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    y = coeff[0]
    for d in range(window):       
        y += coeff[d + 1] * lag[window - d - 1]
        #print(coeff[d + 1] * lag[window - d - 1])
    predictions.append(y)
    history.append(t)
    
mean_squared_error(test,predictions)

plt.plot(test,label='actual')
plt.plot(predictions,label='predicted')
plt.legend()

# on retient un modèle avec lag = 32