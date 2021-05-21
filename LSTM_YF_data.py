import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel(r'E:\Python Projects\Kitt\Data\SP500\SP500.xlsx', index_col=0, parse_dates = True)
date = '2019-01-31'

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
all_data= download_data(sign=latest_rsi, start='2016-01-01', interval='1d',  end='2019-12-31')

def ts_train_test(all_data,time_steps,for_periods):
   
    # create training and test set
    in_sample = 0.8  
    ts_train = all_data[0:int(len(all_data)*in_sample)]
    ts_test  =  all_data[int(len(all_data)*in_sample):]
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)
    
    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps,ts_train_len-1): 
        X_train.append(ts_train[i-time_steps:i])
        y_train.append(ts_train[i:i+for_periods])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
    
    # Preparing to create X_test
    inputs = pd.concat((all_data[0:int(len(all_data)*in_sample)], all_data[int(len(all_data)*in_sample):]),axis=0).values
    inputs = inputs[len(inputs)-len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1,1)
    
    
    X_test = []
    for i in range(time_steps,ts_test_len+time_steps-for_periods):
        X_test.append(inputs[i-time_steps:i,0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    
    return X_train, y_train , X_test

X_train, y_train, X_test = ts_train_test(all_data,5,2)
X_train.shape[0],X_train.shape[1]


X_train_see = pd.DataFrame(np.reshape(X_train, (X_train.shape[0],X_train.shape[1])))
# # y_train_see = pd.DataFrame(y_train)
# pd.concat([X_train_see,y_train_see],axis=1)

# Convert the 3-D shape of X_test to a data frame so we can see: 
X_test_see = pd.DataFrame(np.reshape(X_test, (X_test.shape[0],X_test.shape[1])))
pd.DataFrame(X_test_see)


def ts_train_test(all_data, time_steps,for_periods):
   
    # create training and test set
    in_sample = 0.8  
    ts_train = all_data[0:int(len(all_data)*in_sample)]
    ts_test  =  all_data[int(len(all_data)*in_sample):]
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps,ts_train_len-1): 
        X_train.append(ts_train[i-time_steps:i])
        y_train.append(ts_train[i:i+for_periods])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

    # Preparing to create X_test
    inputs = pd.concat((all_data[0:int(len(all_data)*in_sample)], all_data[int(len(all_data)*in_sample):]),axis=0).values
    inputs = inputs[len(inputs)-len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1,1)

    X_test = []
    for i in range(time_steps,ts_test_len+time_steps-for_periods):
        X_test.append(inputs[i-time_steps:i,0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    return X_train, y_train , X_test

X_train, y_train, X_test = ts_train_test(all_data,5,2)
X_train.shape[0],X_train.shape[1]


X_train_see = pd.DataFrame(np.reshape(X_train, (X_train.shape[0],X_train.shape[1])))
# y_train_see = pd.DataFrame(y_train)
# pd.concat([X_train_see,y_train_see],axis=1)

# Convert the 3-D shape of X_test to a data frame so we can see: 
X_test_see = pd.DataFrame(np.reshape(X_test, (X_test.shape[0],X_test.shape[1])))
pd.DataFrame(X_test_see)


def ts_train_test_normalize(all_data,time_steps,for_periods):
    
    in_sample = 0.8 
    ts_train = all_data[0:int(len(all_data)*in_sample)]
    ts_test  =  all_data[int(len(all_data)*in_sample):]
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    ts_train_scaled = sc.fit_transform(ts_train)

    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps,ts_train_len-1): 
        X_train.append(ts_train_scaled[i-time_steps:i])
        y_train.append(ts_train_scaled[i:i+for_periods])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

    inputs = pd.concat((all_data[0:int(len(all_data)*in_sample)], all_data[int(len(all_data)*in_sample):]),axis=0).values
    inputs = inputs[len(inputs)-len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1,1)
    inputs  = sc.transform(inputs)

    # Preparing X_test
    X_test = []
    for i in range(time_steps,ts_test_len+time_steps-for_periods):
        X_test.append(inputs[i-time_steps:i,0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    return ts_train, ts_test, X_train, y_train , X_test, sc
ts_train, ts_test, X_train, y_train , X_test, sc = ts_train_test_normalize(all_data,5,2)


def LSTM_model_regularization(X_train, y_train, X_test, sc):
    # create a model
   #from keras.models import Sequential
    
    from tensorflow.python.keras.engine.sequential import Sequential
    from keras.layers import Dense, LSTM, Dropout
    
    
    #from keras.optimizers import SGD
    from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
        
    # The LSTM architecture
    my_LSTM_model = Sequential()
    my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    #my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    # my_LSTM_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    my_LSTM_model.add(LSTM(units=50, activation='tanh'))
    my_LSTM_model.add(Dropout(0.2))
    my_LSTM_model.add(Dense(units=2))

    # Compiling
    my_LSTM_model.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
    # Fitting to the training set
    my_LSTM_model.fit(X_train,y_train,epochs=50,batch_size=150, verbose=0)

    LSTM_prediction = my_LSTM_model.predict(X_test)
    LSTM_prediction = sc.inverse_transform(LSTM_prediction)
    trainScore =  my_LSTM_model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f' % trainScore)
    return my_LSTM_model, LSTM_prediction, trainScore

my_LSTM_model, LSTM_prediction, trainScore= LSTM_model_regularization(X_train, y_train, X_test, sc) 


def actual_pred_plot(preds, period):
    actual_pred = pd.DataFrame(columns = ['Close', 'prediction'])
    actual_pred['Close'] = all_data.loc[period:,'Close'][0:len(preds)]
    actual_pred['prediction'] = preds[:,0]

    #from keras.metrics  import MeanSquaredError
    from tensorflow.python.keras.losses import mean_squared_error
    m = mean_squared_error()
    m.update_state(np.array(actual_pred['Close']), np.array(actual_pred['prediction']))
    
    return (m.result().numpy(), actual_pred, actual_pred.plot())

actual_pred_plot(LSTM_prediction, '2019-01') 




#n-step prediction     
last_60 = all_data[-60:].values
last_60_scaled = sc.transform(last_60)
pred_test = []
pred_test.append(last_60_scaled)
pred_test = np.array(pred_test)
pred_test = np.reshape(pred_test, (pred_test.shape[0],pred_test.shape[1], 1))
pred_price = my_LSTM_model.predict(pred_test)
pred_price = sc.inverse_transform(pred_price)
print(pred_price)







    

