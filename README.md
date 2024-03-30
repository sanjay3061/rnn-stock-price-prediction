# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

Problem: Develop an RNN-based model to predict stock prices using historical data.

Dataset: Historical stock price data including date, open, high, low, close prices, volume, and adjusted close.

Steps:

Preprocessing: Clean, normalize, and split data into training/testing sets.

Model Architecture: Design RNN architecture (e.g., LSTM) and configure layers.

Training: Train model on training data using backpropagation through time.

Evaluation: Assess model performance using metrics like Mean Squared Error on testing data.

Prediction: Utilize trained model to forecast future stock prices.

Deployment: Deploy model for real-time prediction, periodically updating with new data.


## Design Steps
### Step 1:
Prepare training data by scaling and creating sequences.
### Step 2:
Add SimpleRNN and Dense layers after initializing a sequential model.
### Step 3:
Use the Adam optimizer and mean squared error loss to compile the model.
### Step 4:
Use the ready-made training data to train the model.
### Step 5:
Use the trained model to make predictions, preprocess test data, and display the outcomes.

## Program
#### Name:Sanjay.R
#### Register Number:212222220038
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('trainset.csv')
dataset_train.columns
dataset_train.head()
train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape
length = 60
n_features = 1

model = Sequential()
model.add(layers.SimpleRNN(50,input_shape=(length,n_features)))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mse')


model.summary()

model.fit(X_train1,y_train,epochs=100, batch_size=32)

dataset_test = pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)



print("Name: Sanjay.R")
print("Register Number: 212222220038")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

```
## Output
![image](https://github.com/sanjay3061/rnn-stock-price-prediction/assets/121215929/eda76976-6b3f-4da5-ab41-4a782ecb24e9)



### Mean Square Error

![image](https://github.com/sanjay3061/rnn-stock-price-prediction/assets/121215929/6c8f1a25-4b83-438f-a553-6a060b7bf1f8)


## Result
Thus a Recurrent Neural Network model for stock price prediction is done.

