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
      ax.set_title(title_lst[i], fontsize=20)
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

def plot_mean_max_min(data):
  '''
  Plots and saves figure showing the mean, max, and min for year, month, day of week and hour.

  INPUT:
      - Pandas Dataframe data
  OUTPUT:
      - Matplotlib Figure
  '''
  fig, axs = plt.subplots(2,2, figsize=(16,10))

  year_mean = data.groupby('Year')['Demand'].mean()
  year_max = data.groupby('Year')['Demand'].max()
  year_min = data.groupby('Year')['Demand'].min()
  year_lst = [year_mean, year_max, year_min]

  months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
  month_mean = data.groupby('Month')['Demand'].mean()
  month_max = data.groupby('Month')['Demand'].max()
  month_min = data.groupby('Month')['Demand'].min()
  month_lst = [month_mean, month_max, month_min]

  days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
  weekday_mean = data.groupby('Day_of_Week')['Demand'].mean()
  weekday_max = data.groupby('Day_of_Week')['Demand'].max()
  weekday_min = data.groupby('Day_of_Week')['Demand'].min()
  weekday_lst = [weekday_mean, weekday_max, weekday_min]

  hour_mean = data.groupby('Hour')['Demand'].mean()
  hour_max = data.groupby('Hour')['Demand'].max()
  hour_min = data.groupby('Hour')['Demand'].min()
  hour_lst = [hour_mean, hour_max, hour_min]

  lst_of_lsts = [year_lst, month_lst, weekday_lst, hour_lst]
  lst_of_index = [year_mean.index, months, days_of_week, hour_mean.index]
  titles = ['Year Mean, Max & Min', 'Month Mean, Max & Min', 'Day of Week Mean, Max & Min', 'Hour Mean, Max & Min']
  x_labels = ['Year', 'Month', 'Day of Week', 'Hour']

  for i,ax in enumerate(axs.flatten()):
    ax.plot(lst_of_index[i], lst_of_lsts[i][0], color='darkgreen')
    ax.plot(lst_of_index[i], lst_of_lsts[i][1], color='lightgreen')
    ax.plot(lst_of_index[i], lst_of_lsts[i][2], color='lightgreen')
    ax.set_title(titles[i], fontsize=20)
    ax.set_xlabel(x_labels[i], fontsize=16)

    if (i==0) | (i==2):
      ax.set_ylabel('Megawatthours', fontsize=16)
    if i==3:
      ax.set_xticks(np.arange(len(hour_mean)))

  plt.tight_layout()
  fig.savefig('images/mean_max_min.png')

def plot_avg_demand_hour(data):
  '''
  Plots and saves figure showing the average demand by hour for four months.

  INPUT:
      - Pandas Dataframe data
  OUTPUT:
      - Matplotlib Figure
  '''
  fig, axs = plt.subplots(figsize=(10,5))

  jan_demand = data[data['Month']==1].groupby(['Hour'])['Demand'].mean()
  apr_demand = data[data['Month']==4].groupby(['Hour'])['Demand'].mean()
  jul_demand = data[data['Month']==7].groupby(['Hour'])['Demand'].mean()
  oct_demand = data[data['Month']==10].groupby(['Hour'])['Demand'].mean()

  axs.plot(jan_demand.index, jan_demand, label='January')
  axs.plot(apr_demand.index, apr_demand, label='April')
  axs.plot(jul_demand.index, jul_demand, label='July')
  axs.plot(oct_demand.index, oct_demand, label='October')

  axs.set_title('Average Demand by Hour by Month', fontsize=20)
  axs.set_ylabel('Megawatthours', fontsize=16)
  axs.set_xlabel('Hour', fontsize=16)
  axs.set_xticks(np.arange(len(jan_demand)))
  axs.legend(loc='best')

  plt.tight_layout()
  fig.savefig('images/avg_hour_month.png')