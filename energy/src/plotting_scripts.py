import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_series(raw_data, data):
  '''
  Plots and saves figure showing time series data before and after dealing with outliers.

  INPUT:
      - Pandas Dataframe raw_data
      - Pandas Dataframe data
  OUTPUT:
      - Matplotlib Figure
  '''
  fig, axs = plt.subplots(1,2, figsize=(14, 5))

  data_lst = [raw_data, data]
  title_lst = ['Hourly Demand- Outliers Unaltered', 'Hourly Demand- Outliers Altered']

  for i,ax in enumerate(axs.flatten()):
      ax.plot(data_lst[i]['Timestamp'], data_lst[i]['Demand'])
      ax.set_title(title_lst, fontsize=20)
      ax.set_xlabel('Time', fontsize=16)

      if i==0:
          ax.set_ylabel('Megawatthours', fontsize=16)

  plt.tight_layout()
  fig.savefig('images/series.png')

def plot_decomposed_series(data):
  '''
  Plots and saves figure showing trend, seasonal, and residual components.

  INPUT:
      - Pandas Dataframe data
  OUTPUT:
      - Matplotlib Figure
  '''
  fig, axs = plt.subplots(2,2, figsize=(14,10), sharex=True)

  df = data.set_index('Timestamp')
  demand_decomposition = sm.tsa.seasonal_decompose(df['Demand'], period=24*7*52)
  titles = ['Hourly Demand', 'Trend Component', 'Seasonal Component', 'Residual Component']
  series_lst = [df['Demand'], demand_decomposition.trend, demand_decomposition.seasonal, demand_decomposition.resid]

  for i,ax in enumerate(axs.flatten()):
    ax.plot(data['Timestamp'], series_lst[i])
    ax.set_title(titles[i], fontsize=20)

    if i == 1:
      ax.set_ylim(15000,65000)
    if (i==0) | (i==2):
      ax.set_ylabel('Megawatthours', fontsize=16)

  plt.tight_layout()
  fig.savefig('images/decomposed.png')