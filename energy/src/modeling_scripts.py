import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def train_test_data(data, pred_date, weeks=1, days=1, hours=24):
  '''
  Splits data into training and test dataframes depending on date given to predict after and time to predict.

  INPUT:
      - Pandas Dataframe data
      - String pred_date
      - Int weeks, days and hours (multipled together will give total hours to predict)
  OUTPUT:
      - Pandas Dataframe train_data
      - Pandas Dataframe test_data
  '''
  train_data = data[data['Timestamp'] < pred_date]
  end_pred = weeks * days * hours
  test_data = data[data['Timestamp'] >= pred_date]
  return train_data, test_data[:end_pred]

def prep_data_basic_models(dum_data, pred_date, weeks=1, days=1, hours=24):
  '''
  Splits data into training and test dataframes depending on date given to predict after and time to predict.
  Splits those dataframes into features and targets.

  INPUT:
      - Pandas Dataframe dum_data
      - String pred_date
      - Int weeks, days and hours (multipled together will give total hours to predict)
  OUTPUT:
      - Pandas Dataframe X_train
      - Pandas Series y_train
      - Pandas Dataframe X_test
      - Pandas Series y_test
  '''
  data_train, data_test = train_test_data(dum_data, pred_date, weeks, days, hours)
  y_train = data_train.pop('Demand')
  X_train = data_train
  del X_train['Timestamp']
  y_test = data_test.pop('Demand')
  X_test = data_test
  del X_test['Timestamp']
  return X_train, y_train, X_test, y_test

def score_avg_model(data, pred_date, weeks=1, days=1, hours=24):
  '''
  Uses root mean squared error to score model that takes the average of time period before the same length as period to be predicted.

  INPUT:
      - Pandas Dataframe data
      - String pred_date
      - Int weeks, days and hours (multipled together will give total hours to predict)
  OUTPUT:
      - Float
  '''
  train_data, test_data = train_test_data(data, pred_date, weeks=weeks, days=days, hours=hours)
  y_test = test_data['Demand']
  avg = train_data['Demand'][-len(y_test):].mean()
  yhat = np.full(shape=(len(y_test)), fill_value = avg)
  return mean_squared_error(y_test, yhat, squared=False)

def score_last_period_model(data, pred_date, weeks=1, days=1, hours=24):
  '''
  Uses root mean squared error to score model that uses previous time period demand to predict future demand.

  INPUT:
      - Pandas Dataframe data
      - String pred_date
      - Int weeks, days and hours (multipled together will give total hours to predict)
  OUTPUT:
      - Float
  '''
  train_data, test_data = train_test_data(data, pred_date, weeks=weeks, days=days, hours=hours)
  y_test = test_data['Demand']
  yhat = train_data['Demand'][-len(y_test):]
  return mean_squared_error(y_test, yhat, squared=False)

def score_last_year_model(data, pred_date, weeks=1, days=1, hours=24):
  '''
  Uses root mean squared error to score model using demand for the same time last year.

  INPUT:
      - Pandas Dataframe data
      - String pred_date
      - Int weeks, days and hours (multipled together will give total hours to predict)
  OUTPUT:
      - Float
  '''
  train_data, test_data = train_test_data(data, pred_date, weeks=weeks, days=days, hours=hours)
  y_test = test_data['Demand']
  start = 52*7*24
  stop = weeks*days*hours
  yhat = train_data[-start:-start+stop]['Demand']
  return mean_squared_error(y_test, yhat, squared=False)

def get_predictions(model, dum_data, pred_date, weeks=1, days=1, hours=1):
  '''
  Returns arrays of predictions and actual values given a model, dummified data, prediction date and length of time to predict.

  INPUT:
      - Model Class model
      - Pandas Dataframe dum_data
      - String pred_date
      - Int weeks, days and hours (multipled together will give total hours to predict)
  OUTPUT:
      - Array yhat
      - Array y_test
  '''
  X_train, y_train, X_test, y_test = prep_data_basic_models(dum_data, pred_date, weeks=weeks, days=days, hours=hours)
  fitted_model = model.fit(X_train, y_train)
  yhat = fitted_model.predict(X_test)
  return yhat, y_test

def score_basic_models(model, dum_data, dates_lst, weeks=1, days=1, hours=1):
  '''
  Uses root mean squared error to score model on five different dates. Returns a list of those scores.

  INPUT:
      - Model Class model
      - Pandas Dataframe dum_data
      - List of Strings dates_lst
      - Int weeks, days and hours (multipled together will give total hours to predict)
  OUTPUT:
      - List of Floats rmse
  '''
  rmse = []
  for date in dates_lst:
    yhat, y_test = get_predictions(model, dum_data, date, weeks=weeks, days=days, hours=hours)
    rmse.append(mean_squared_error(y_test, yhat, squared=False))
  return rmse

def score_basic_models_test(model, dum_data, pred_date, weeks=1, days=1, hours=1):
  '''
  Uses root mean squared error to score model on final testing data.

  INPUT:
      - Model Class model
      - Pandas Dataframe dum_data
      - String pred_date
      - Int weeks, days and hours (multipled together will give total hours to predict)
  OUTPUT:
      - Float rmse
  '''
  yhat, y_test = get_predictions(model, dum_data, pred_date, weeks=weeks, days=days, hours=hours)
  rmse = mean_squared_error(y_test, yhat, squared=False)
  return rmse

def windowize_data(data, window_size, num_pred):
  '''
  Windowize data set. Returns X array of with window_size data points and y array with num_pred data points.

  INPUT:
      - Pandas Dataframe data
      - Int window_size
      - Int num_pred
  OUTPUT:
      - Array X
      - Array y
  '''
  X, y = [], []
  for i in range(len(data)):
    if window_size + i + num_pred < len(data)+1:
      x_values = data['Normalized'][i:window_size+i]
      y_values = data['Normalized'][window_size+i:num_pred+window_size+i]
      X.append(x_values)
      y.append(y_values)
    else:
      break
  return np.array(X), np.array(y)

def prep_data_lstm(data, pred_date, window_size, weeks=1, days=7, hours=24):
  '''
  Splits data into X_train, y_train, X_test, y_test and converts them into right shape for LSTM model.

  INPUT:
      - Pandas Dataframe data
      - String pred_date
      - Int window_size
      - Int weeks, days and hours (multipled together will give total hours to predict)
  OUTPUT:
      - Matrix X_train
      - Array y_train
      - Matrix X_test
      - Array y_test
  '''

  num_pred = weeks * days * hours

  train_data, test_data = train_test_data(data, pred_date, weeks=1, days=7, hours=24)
  all_data = train_data.merge(test_data, how='outer')

  X, y = windowize_data(all_data, window_size, num_pred)
  X_train, y_train = X[:len(X)-1-num_pred], y[:len(y)-1-num_pred]
  X_test, y_test = X[-1], y[-1]
  return X_train[:,:,None], y_train, X_test.reshape(1,-1)[:,:,None], y_test.reshape(1,-1)