# Import libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os
from sklearn.model_selection import GridSearchCV
import pickle


f = open('NN.pickle', 'rb')
mlp = pickle.load(f)
priceData = pd.read_csv('S&P_test.csv')
dataset = priceData[['Date', 'Open', 'High', 'Low', 'Close']]
dataset = dataset.dropna()

# Add financial indicators to dataset
dataset['H-L'] = dataset['High'] - dataset['Low']
dataset['O-C'] = dataset['Close'] - dataset['Open']
dataset['3day SMA'] = dataset['Close'].shift(1).rolling(window = 3).mean()
dataset['10day SMA'] = dataset['Close'].shift(1).rolling(window = 10).mean()
dataset['20day SMA'] = dataset['Close'].shift(1).rolling(window = 20).mean()
# dataset['30day SMA'] = dataset['Close'].shift(1).rolling(window = 30).mean()
dataset['Std_dev'] = dataset['Close'].rolling(5).std()
dataset['5day EMA'] = dataset['Close'].ewm(span=5, adjust=False).mean() #short EMA
dataset['50day EMA'] = dataset['Close'].ewm(span=50, adjust=False).mean() #long EMA
# dataset['20day EMA'] = dataset['Close'].ewm(span=20, adjust=False).mean() #long EMA
dataset['Bollinger Percent'] = (dataset['Close'] - (dataset['20day SMA']-(2*dataset['Close'].rolling(20).std())))/((dataset['20day SMA']+(2*dataset['Close'].rolling(20).std())) - (dataset['20day SMA']-(2*dataset['Close'].rolling(20).std())))
dataset['Momentum'] = dataset['Close'] - dataset['Close'].shift(-10)
dataset = dataset.dropna()

modelData = dataset.iloc[:, dataset.columns != 'Date']
# Pass data into model
pricePred = mlp.predict(modelData)

dataset['Price Prediction'] = np.NaN
dataset.iloc[(len(dataset) - len(pricePred)):,-1:] = pricePred

print(dataset)

# Import data
sentData = pd.read_csv('wsjSentiment.csv')
sentData = sentData.dropna()


joinedData = pd.merge(dataset, sentData, how='outer', left_on = "Date", right_on = "Date")
#print(joinedData)
joinedData.to_csv('JoinedData.csv')
f.close()

trade_dataset = joinedData.dropna()

trade_dataset['Tomorrows Returns'] = 0.
trade_dataset['Tomorrows Returns'] = np.log(trade_dataset['Close']/trade_dataset['Close'].shift(1))
trade_dataset['Tomorrows Returns'] = trade_dataset['Tomorrows Returns'].shift(-1)

trade_dataset['Strategy Returns'] = 0.

# Assumes buy and sell
trade_dataset['Strategy Returns'] = np.where((trade_dataset['Price Prediction'] == 1) | (trade_dataset['Sentiment'] == "pos"), trade_dataset['Tomorrows Returns'], -trade_dataset['Tomorrows Returns'])


# Assumes buy and hold
#trade_dataset['Strategy Returns'] = np.where((trade_dataset['Price Prediction'] == 1) | ##(trade_dataset['Sentiment'] == "pos"), trade_dataset['Tomorrows Returns'], 0)

trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Tomorrows Returns'])
trade_dataset['Cumulative Strategy Returns'] = np.cumsum(trade_dataset['Strategy Returns'])

import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.legend()
plt.show()
