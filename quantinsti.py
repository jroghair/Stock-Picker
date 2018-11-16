import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
from stockstats import StockDataFrame as Sdf

import random
random.seed(42)

dataset = pd.read_csv('S&P.csv')
dataset = dataset.dropna()
dataset = dataset[['Open', 'High', 'Low', 'Close']]



dataset['H-L'] = dataset['High'] - dataset['Low']
dataset['O-C'] = dataset['Close'] - dataset['Open']
dataset['3day MA'] = dataset['Close'].shift(1).rolling(window = 3).mean()
dataset['10day MA'] = dataset['Close'].shift(1).rolling(window = 10).mean()
dataset['30day MA'] = dataset['Close'].shift(1).rolling(window = 30).mean()
dataset['Std_dev']= dataset['Close'].rolling(5).std()
print(dataset.head())

dataset['RSI'] = Sdf.retype(dataset)['rsi_12']
del dataset['close_-1_s']
del dataset[ 'close_-1_d']
del dataset['rs_12']
del dataset['rsi_12']

print(dataset.head())
print(list(dataset)) # list of all column names

#print (dataset.loc[0:5, ['3day MA', 'Close']])


dataset['Price_Rise'] = np.where(dataset['close'].shift(-1) > dataset['close'], 1, 0) #1 when the closing price of tomorrow is greater than the closing price of today.

#print(dataset.loc[0:10, ['close', 'Price_Rise']])


dataset = dataset.dropna()

# We then create two data frames storing the input and the output variables. The dataframe ‘X’ stores the input features, the columns starting from the fifth column (or index 4) of the dataset till the second last column. The last column will be stored in the dataframe y, which is the value we want to predict, i.e. the price rise.
X = dataset.iloc[:, 4:-1]
#print(X)
y = dataset.iloc[:, -1]

#print(y)


split = int(len(dataset)*0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:] 


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#print(X_train)


classifier = Sequential()
# First layer
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
# Second layer
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
# Output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# COMPILE
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


y_pred = classifier.predict(X_test)
#print("first", y_pred)
y_pred = (y_pred > 0.5)
#print("second", y_pred)

dataset['y_pred'] = np.NaN
dataset.iloc[(len(dataset) - len(y_pred)):,-1:] = y_pred
trade_dataset = dataset.dropna()



trade_dataset['Tomorrows Returns'] = 0.
trade_dataset['Tomorrows Returns'] = np.log(trade_dataset['close']/trade_dataset['close'].shift(1))
trade_dataset['Tomorrows Returns'] = trade_dataset['Tomorrows Returns'].shift(-1)

trade_dataset['Strategy Returns'] = 0.
trade_dataset['Strategy Returns'] = np.where(trade_dataset['y_pred'] == True, trade_dataset['Tomorrows Returns'], - trade_dataset['Tomorrows Returns'])

trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Tomorrows Returns'])
trade_dataset['Cumulative Strategy Returns'] = np.cumsum(trade_dataset['Strategy Returns'])

print("dataset")
print(dataset.tail())
print(trade_dataset.tail())
print(trade_dataset.iloc[:, 6:12])

plt.figure(figsize=(10,5))
plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.legend()
plt.show()














