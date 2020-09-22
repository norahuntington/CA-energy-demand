import pandas as pd
import numpy as np
import requests

def make_and_save_dataframe():
    '''
        Connects to API. Gets data. Extracts year, month, day of week, and hour to make new columns.
        Saves data to csv file in data folder.

        INPUT:
            - None
        OUTPUT:
            - Saved csv file containing pandas dataframe
    '''
    request = requests.get('http://api.eia.gov/series/?api_key=YOURKEYHERE&series_id=EBA.CAL-ALL.D.HL')
    data_json = request.json()
    data = pd.DataFrame(data_json['series'][0]['data'], columns=['Timestamp', 'Demand'])
    data['Timestamp'] = data['Timestamp'].str.split('-', expand=True)[0]
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Year'] = data['Timestamp'].dt.year
    data['Month'] = data['Timestamp'].dt.month
    data['Day_of_Week'] = data['Timestamp'].dt.weekday
    data['Hour'] = data['Timestamp'].dt.hour
    clean_data = data.sort_values('Timestamp').reset_index(drop=True)
    clean_data.to_csv('data/ca_energy_demand.csv', index=False)

def clean_data(data):
    '''
        Finds outliers and replaces with average for same month, day of week, and hour.

        INPUT:
            - Pandas dataframe
        OUTPUT:
            - Pandas dataframe
    '''
    clean_data = data.copy()
    outliers = data[data['Demand'] < 15000]
    for i,ind in enumerate(outliers.index):
        sim_points = data[(data['Month']==outliers['Month'].loc[ind]) & (data['Day_of_Week']==outliers['Day_of_Week'].loc[ind]) & (data['Hour']==outliers['Hour'].loc[ind])]
        avg_sim_points = (sum(sim_points['Demand'])-outliers['Demand'].iloc[i])/(len(sim_points)-1)
        clean_data['Demand'].iloc[ind] = avg_sim_points
    return clean_data

def prep_dum_data(data):
    '''
        Dummifies year, month, day of week, and hour columns.

        INPUT:
            - Pandas dataframe
        OUTPUT:
            - Pandas dataframe
    '''
    month_names = {2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    dum_month = pd.get_dummies(data['Month'], drop_first=True)
    dum_month = dum_month.rename(columns=month_names)
    weekday_names = {1:'Tues',2:'Wed',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}
    dum_weekday = pd.get_dummies(data['Day_of_Week'], drop_first=True)
    dum_weekday = dum_weekday.rename(columns=weekday_names)
    dum_hour = pd.get_dummies(data['Hour'], drop_first=True)
    dum_year = pd.get_dummies(data['Year'], drop_first=True)
    sub_data = data[['Timestamp','Demand']]
    dum_data = dum_month.join([sub_data, dum_weekday, dum_hour, dum_year], how='outer')
    return dum_data

def normalize(data):
    '''
        Normalizes demand.

        INPUT:
            - Pandas dataframe
        OUTPUT:
            - Pandas dataframe
    '''
    max_demand = max(data['Demand'])
    min_demand = min(data['Demand'])
    data['Normalized'] = (data['Demand']-min_demand) / ((max_demand - min_demand)*10)
    return data