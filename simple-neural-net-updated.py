# Import libraries
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import os

# Import data
os.chdir('C:\\Users\\Jeremy\\572 Data\Stock-Picker')

# Import data
dataset = pd.read_csv('S&P_68yrs.csv')
dataset = dataset.dropna()
dataset = dataset[['Open', 'High', 'Low', 'Close']]

# Add financial indicators to dataset
dataset['H-L'] = dataset['High'] - dataset['Low']
dataset['O-C'] = dataset['Close'] - dataset['Open']
dataset['3day MA'] = dataset['Close'].shift(1).rolling(window = 3).mean()
dataset['10day MA'] = dataset['Close'].shift(1).rolling(window = 10).mean()
dataset['30day MA'] = dataset['Close'].shift(1).rolling(window = 30).mean()
dataset['Std_dev']= dataset['Close'].rolling(5).std()
dataset['Price_Rise'] = np.where(dataset['Close'].shift() < dataset['Close'], 1, 0)

#dataset.to_csv('testoutput.csv')

dataset = dataset.dropna()

# Dataset storing input variables
X = dataset.iloc[:, :-1]
# Dataset storing output variable Price_Rise
y = dataset.iloc[:, -1]

# print(list(y.columns.values))

# Split data into training (80%) and testing set (20%)
split = int(len(dataset)*0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Build NN
mlp = MLPClassifier(alpha=1, random_state=0, activation='relu', solver='adam', hidden_layer_sizes=[50, 25])
mlp.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# Make predictions and evaluate the model
y_pred = mlp.predict(X_test)
#print(y_test)
print(y_pred)

accuracy = mlp.score(X_test_scaled, y_test)
print("The accuracy of this model is: ", accuracy*100, " %")


# Compare strategy and market returns
dataset['y_pred'] = np.NaN
dataset.iloc[(len(dataset) - len(y_pred)):,-1:] = y_pred
trade_dataset = dataset.dropna()

trade_dataset['Tomorrows Returns'] = 0.
trade_dataset['Tomorrows Returns'] = np.log(trade_dataset['Close']/trade_dataset['Close'].shift(1))
trade_dataset['Tomorrows Returns'] = trade_dataset['Tomorrows Returns'].shift(-1)

trade_dataset['Strategy Returns'] = 0.
trade_dataset['Strategy Returns'] = np.where(trade_dataset['y_pred'] == True, trade_dataset['Tomorrows Returns'], - trade_dataset['Tomorrows Returns'])

trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Tomorrows Returns'])
trade_dataset['Cumulative Strategy Returns'] = np.cumsum(trade_dataset['Strategy Returns'])

import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.legend()
plt.show()

trade_dataset.to_csv('testoutput.csv')

print('working?')