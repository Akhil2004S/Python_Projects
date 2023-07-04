def crypto_prediction(choice,year,month,day):

  import json
  import requests
  from keras.models import Sequential
  from keras.layers import Activation, Dense, Dropout, LSTM
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  import seaborn as sns
  from sklearn.metrics import mean_absolute_error
  from keras.preprocessing.sequence import TimeseriesGenerator
  from datetime import date, datetime,timedelta
  print(type(year))

  endpoint = 'https://min-api.cryptocompare.com/data/histoday'
  int_year = int(year)
  int_month = int(month)
  int_day = int(day)

  res = requests.get(endpoint + f'?fsym={choice}&tsym=CAD&limit=2000')
  hist1 = pd.DataFrame(json.loads(res.content)['Data'])
  hist1 = hist1.set_index('time')
  hist1.index = pd.to_datetime(hist1.index, unit='s')
  target_col = 'close'

  d = date(int_year, int_month, int_day)

  hist1.drop(["conversionType", "conversionSymbol"], axis = 'columns', inplace = True)

  hist1.head(10000)


  hist = hist1.iloc[5:2000]

  hist.head(10000)

  def train_test_split(df, test_size=0.2):
      split_row = len(df) - int(test_size * len(df))
      train_data = df.iloc[:split_row]
      test_data = df.iloc[split_row:]
      return train_data, test_data

  train, test = train_test_split(hist, test_size=0.2)

  def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
      fig, ax = plt.subplots(1, figsize=(13, 10))
      ax.plot(line1, label=label1, linewidth=lw)
      ax.plot(line2, label=label2, linewidth=lw)
      ax.set_ylabel('price [CAD]', fontsize=14)
      ax.set_title(title, fontsize=16)
      ax.legend(loc='best', fontsize=16);

  #line_plot(train[target_col], test[target_col], 'training', 'test', title='')

  close_data = hist['close'].values
  close_data = close_data.reshape((-1,1))

  split_percent = 0.80
  split = int(split_percent*len(close_data))

  close_train = close_data[:split]
  close_test = close_data[split:]

  date_train = hist.index[:split]
  date_test = hist.index[split:]

  look_back = 15

  train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
  test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

  def normalise_zero_base(df):
      return df / df.iloc[0] - 1

  def normalise_min_max(df):
      return (df - df.min()) / (data.max() - df.min())

  def extract_window_data(df, window_len=5, zero_base=True):
      window_data = []
      for idx in range(len(df) - window_len):
          tmp = df[idx: (idx + window_len)].copy()
          if zero_base:
              tmp = normalise_zero_base(tmp)
          window_data.append(tmp.values)
      return np.array(window_data)

  def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
      train_data, test_data = train_test_split(df, test_size=test_size)
      X_train = extract_window_data(train_data, window_len, zero_base)
      X_test = extract_window_data(test_data, window_len, zero_base)
      y_train = train_data[target_col][window_len:].values
      y_test = test_data[target_col][window_len:].values
      if zero_base:
          y_train = y_train / train_data[target_col][:-window_len].values - 1
          y_test = y_test / test_data[target_col][:-window_len].values - 1

      return train_data, test_data, X_train, X_test, y_train, y_test

  def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear',
                      dropout=0.2, loss='mse', optimizer='adam'):
      model = Sequential()
      model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
      model.add(Dropout(dropout))
      model.add(Dense(units=output_size))
      model.add(Activation(activ_func))

      model.compile(loss=loss, optimizer=optimizer)
      return model

  np.random.seed(42)
  window_len = 5
  test_size = 0.2
  zero_base = True
  lstm_neurons = 100
  epochs = 20
  batch_size = 32
  loss = 'mse'
  dropout = 0.2
  optimizer = 'adam'

  train, test, X_train, X_test, y_train, y_test = prepare_data(
      hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

  model = build_lstm_model(
      X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
      optimizer=optimizer)
  history = model.fit(
      X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

  targets = test[target_col][window_len:]
  preds = model.predict(X_test).squeeze()
  mean_absolute_error(preds, y_test)

  from sklearn.metrics import mean_squared_error
  MAE=mean_squared_error(preds, y_test)
  MAE

  from sklearn.metrics import r2_score
  R2=r2_score(y_test, preds)
  R2

  preds = test[target_col].values[:-window_len] * (preds + 1)
  preds = pd.Series(index=targets.index, data=preds)
  line_plot(targets, preds, 'actual', 'prediction', lw=3)

  model = Sequential()
  model.add(
      LSTM(10,
          activation='relu',
          input_shape=(look_back,1))
  )
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')

  num_epochs = 25
  model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

  close_data = close_data.reshape((-1))


  def predict(num_prediction, model):
      prediction_list = close_data[-look_back:]

      for _ in range(num_prediction):
          x = prediction_list[-look_back:]
          x = x.reshape((1, look_back, 1))
          out = model.predict(x)[0][0]
          prediction_list = np.append(prediction_list, out)
      prediction_list = prediction_list[look_back - 1:]

      return prediction_list


  def predict_dates(num_prediction):
      last_date = hist.index.values[-1]
      prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
      return prediction_dates


  num_prediction = 30
  forecast = predict(num_prediction, model)
  forecast_dates = predict_dates(num_prediction)
  def plot():
    fig, ax = plt.subplots(1, figsize=(13, 10))
    ax.plot(train[target_col], label='Past')
    ax.plot(test[target_col], label='Testing')
    ax.plot(forecast_dates,forecast,label='Prediction')
    ax.set_ylabel('price [CAD]', fontsize=14)
    ax.set_title('title', fontsize=16)
    ax.legend(loc='best', fontsize=16);
  d = date(int_year, int_month , int_day)
  print(d)

  prediction_dates=[]
  for i in range(0,30):
    var=date.today()+timedelta(days=i)
    prediction_dates.append(var.strftime("%Y-%m-%d"))
  print(prediction_dates)
  for i in range(0,30):
    if(prediction_dates[i] == str(d)):
      a=forecast[i]
      return a

# Uncomment the code below to test the function
# value = crypto_prediction('doge', 2023.0, 1.0, 25.0)
# print(value)