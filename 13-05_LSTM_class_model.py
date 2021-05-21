import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import tensorflow 
from tensorflow.keras import Sequential
from keras.layers import LSTM, Dense, Dropout
import requests


#LSTM class
class DeepModelTS():
    """
    A class to create a deep time series model
    """
    def __init__(
        self, 
        data: pd.DataFrame, 
        Y_var: str,
        lag: int,
        LSTM_layer_depth: int, 
        epochs=10, 
        batch_size=256,
        train_test_split=0
    ):

        self.data = data 
        self.Y_var = Y_var 
        self.lag = lag 
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split

    @staticmethod
    def create_X_Y(ts: list, lag: int) -> tuple:
        """
        A method to create X and Y matrix from a time series list for the training of 
        deep learning models 
        """
        X, Y = [], []

        if len(ts) - lag <= 0:
            X.append(ts)
        else:
            for i in range(len(ts) - lag):
                Y.append(ts[i + lag])
                X.append(ts[i:(i + lag)])

        X, Y = np.array(X), np.array(Y)

        # Reshaping the X array to an LSTM input shape 
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, Y         

    def create_data_for_NN(
        self,
        use_last_n=None
        ):
        """
        A method to create data for the neural network model
        """
        # Extracting the main variable we want to model/forecast
        y = self.data[self.Y_var].tolist()

        # Subseting the time series if needed
        if use_last_n is not None:
            y = y[-use_last_n:]

        # The X matrix will hold the lags of Y 
        X, Y = self.create_X_Y(y, self.lag)

        # Creating training and test sets 
        X_train = X
        X_test = []

        Y_train = Y
        Y_test = []

        if self.train_test_split > 0:
            index = round(len(X) * self.train_test_split)
            X_train = X[:(len(X) - index)]
            X_test = X[-index:]     
            
            Y_train = Y[:(len(X) - index)]
            Y_test = Y[-index:]

        return X_train, X_test, Y_train, Y_test

    def LSTModel(self):
        """
        A method to fit the LSTM model 
        """
        # Getting the data 
        X_train, X_test, Y_train, Y_test = self.create_data_for_NN()

        # Defining the model
        model = Sequential()
        model.add(LSTM(self.LSTM_layer_depth, activation='relu', input_shape=(self.lag, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
      

        # Defining the model parameter dict 
        keras_dict = {
            'x': X_train,
            'y': Y_train,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'shuffle': False
        }

        if self.train_test_split > 0:
            keras_dict.update({
                'validation_data': (X_test, Y_test)
            })

        # Fitting the model 
        model.fit(
            **keras_dict
        )

        # Saving the model to the class 
        self.model = model

        return model

    def predict(self) -> list:
        """
        A method to predict using the test data used in creating the class
        """
        yhat = []

        if(self.train_test_split > 0):
        
            # Getting the last n time series 
            _, X_test, _, _ = self.create_data_for_NN()        

            # Making the prediction list 
            yhat = [y[0] for y in self.model.predict(X_test)]

        return yhat

    def predict_n_ahead(self, n_ahead: int):
        """
        A method to predict n time steps ahead
        """    
        X, _, _, _ = self.create_data_for_NN(use_last_n=self.lag)        

        # Making the prediction list 
        yhat = []

        for _ in range(n_ahead):
            # Making the prediction
            fc = self.model.predict(X)
            yhat.append(fc)

            # Creating a new input matrix for forecasting
            X = np.append(X, fc)

            # Ommiting the first variable
            X = np.delete(X, 0)

            # Reshaping for the next iteration
            X = np.reshape(X, (1, len(X), 1))

        return yhat    
    
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from datetime import datetime, timedelta
import yfinance as yf

df = pd.read_excel(r'E:\Python Projects\Kitt\Data\SP500\SP500.xlsx', index_col=0, parse_dates = True)
date = date = '2020-12-31'

def RSI(data, time_window, date=date):    
    diff = data.diff().dropna()     
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    up_chg[diff > 0] = diff[ diff>0 ]
    
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    latest_rsi = rsi.loc[date].idxmax()
    return latest_rsi

def download_data(sign=None, start=None, interval=None, end=None):
  data = yf.download(sign, start=start, interval=interval,  end=end ,progress=False)[['Close']]
  return data

latest_rsi = RSI(df, 5)

start = '2017-01-01'
end = '2020-12-31'
interval = '1d'
unit='d'
all_data = download_data(sign=latest_rsi, start=start, interval=interval,  end=end).reset_index()
d = all_data.rename(columns= {'Close': 'close', 'index':'Date'})
d['Date']= pd.to_datetime(d['Date'], unit=unit)

conf = {'train_test_split': 0.33, 
        'lag': 24,
        'LSTM_layer_depth': 64,
        'epochs': 50}

deep_learner = DeepModelTS(
    data=d, 
    Y_var='close',
    lag=conf.get('lag'),
    LSTM_layer_depth=conf.get('LSTM_layer_depth'),
    epochs=conf.get('epochs'),
    train_test_split=conf.get('train_test_split') # The share of data that will be used for validation
)

# Fitting the model 
model = deep_learner.LSTModel()
yhat = deep_learner.predict()

if len(yhat) > 0:

    # Constructing the forecast dataframe
    fc = d.tail(len(yhat)).copy()
    fc.reset_index(inplace=True)
    fc['forecast'] = yhat

    # Ploting the forecasts
    plt.figure(figsize=(12, 8))
    for dtype in ['close', 'forecast']:
        plt.plot(
            'Date',
            dtype,
            data=fc,
            label=dtype,
            alpha=0.8
        )
    plt.legend()
    plt.grid()
    plt.show()   


X_train, X_test, Y_train, Y_test = deep_learner.create_data_for_NN()
testScore = math.sqrt(mean_squared_error(Y_test, fc['forecast']))
print('RSME of Test: %.2f' % (testScore))