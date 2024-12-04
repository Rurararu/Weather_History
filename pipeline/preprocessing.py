import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from statsmodels.imputation import mice
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pickle
# for Q-Q plots
import scipy.stats as stats
import columns 
import warnings
warnings.simplefilter('ignore')

def preprocess_train_data(ds: pd.DataFrame) -> pd.DataFrame:
    
    ds.dropna(axis=0, inplace=True)
    
    def find_skewed_boundaries(df, variable, distance):

        IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

        lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
        upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

        return upper_boundary, lower_boundary
    
    outliers_left_columns = dict()
    for column in columns.outliers_left_columns:
        if column in ds.columns:
            upper_boundary, lower_boundary = find_skewed_boundaries(ds, column, 2)
            median_value = ds[column].median()
            ds.loc[ds[column] < lower_boundary, column] = median_value
        else:
            print(f"Warning: Column '{column}' not found in the dataset.")


    outliers_right_columns = dict()
    for column in columns.outliers_right_columns:
        upper_boundary, lower_boundary = find_skewed_boundaries(ds, column, 2)
        median_value = ds[column].median()
        ds.loc[ds[column] > upper_boundary, column] = median_value
        outliers_right_columns[column] = median_value
    
    ds.drop(columns.no_need_columns, axis=1)
    
    ds_summary = pd.read_csv('D:/3Kurs/1Sem/SS/Practice/rgr/data/weatherHistory_Summary1.csv')
    
        # Extract unique weather conditions
    conditions = set()
    for summary in ds_summary['Summary']:
        parts = summary.replace(' and ', ', ').split(', ')
        conditions.update(parts)

    # Create binary columns for each condition
    for condition in conditions:
        ds[condition] = ds_summary['Summary'].apply(lambda x: 1 if condition in x else 0)
        
    
    ds['Precip_Rain'] = ds['Precip_Type'].apply(lambda x: 1 if x == 'rain' else 0)
    ds['Snow'] = ds['Precip_Type'].apply(lambda x: 1 if x == 'snow' else 0)

    ds['Rain'] = ds[['Rain', 'Precip_Rain']].sum(axis=1).clip(upper=1)

    ds = ds.drop(columns=[ 'Precip_Rain'])
    
    ds['Formatted_Date'] = pd.to_datetime(ds['Formatted_Date'], errors='coerce', utc=True)

    # Convert the "Formatted_Date" column to datetime
    ds['Formatted_Date'] = pd.to_datetime(ds['Formatted_Date'], format='%Y-%m-%d %H:%M:%S.%f %z')

    # Extract year, month, day, and hour in seconds
    ds['Year'] = ds['Formatted_Date'].dt.year
    ds['Month'] = ds['Formatted_Date'].dt.month
    ds['Hour'] = ds['Formatted_Date'].dt.hour
    
    ds.loc[ds['Year'] == 2005, 'Year'] = 2006

    scaler = MinMaxScaler()
    ds[columns.columns_to_scale] = scaler.fit_transform(ds[columns.columns_to_scale])
    
    ds = ds.drop(columns.drop_columns, axis=1)
    
    param_dict ={
                  'outliers_left_columns':outliers_left_columns,
                  'outliers_right_columns':outliers_right_columns
    }

    with open('D:/3Kurs/1Sem/SS/Practice/rgr/pipeline/param_dict.pickle', 'wb') as handle:
        pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # ds.to_csv('D:/3Kurs/1Sem/SS/Practice/rgr/data/train_look.csv', index=False)
    
    return ds


def preprocess_testing_data(ds: pd.DataFrame) -> pd.DataFrame:
    
    with open('D:/3Kurs/1Sem/SS/Practice/rgr/pipeline/param_dict.pickle', 'rb') as handle:
        param_dict = pickle.load(handle)
    
    ds.dropna(axis=0, inplace=True)
    
    def find_skewed_boundaries(df, variable, distance):

        IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

        lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
        upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

        return upper_boundary, lower_boundary
    
    for column in columns.outliers_left_columns:
        if column in ds.columns:
            upper_boundary, lower_boundary = find_skewed_boundaries(ds, column, 2)
            median_value = ds[column].median()
            ds.loc[ds[column] < lower_boundary, column] = median_value
        else:
            print(f"Warning: Column '{column}' not found in the dataset.")


    for column in columns.outliers_right_columns:
        upper_boundary, lower_boundary = find_skewed_boundaries(ds, column, 2)
        median_value = ds[column].median()
        ds.loc[ds[column] > upper_boundary, column] = median_value
    
    ds.drop(columns.no_need_columns, axis=1)
    
    ds_summary = pd.read_csv('D:/3Kurs/1Sem/SS/Practice/rgr/data/weatherHistory_Summary1.csv')
    
        # Extract unique weather conditions
    conditions = set()
    for summary in ds_summary['Summary']:
        parts = summary.replace(' and ', ', ').split(', ')
        conditions.update(parts)

    # Create binary columns for each condition
    for condition in conditions:
        ds[condition] = ds_summary['Summary'].apply(lambda x: 1 if condition in x else 0)
        
    
    ds['Precip_Rain'] = ds['Precip_Type'].apply(lambda x: 1 if x == 'rain' else 0)
    ds['Snow'] = ds['Precip_Type'].apply(lambda x: 1 if x == 'snow' else 0)

    ds['Rain'] = ds[['Rain', 'Precip_Rain']].sum(axis=1).clip(upper=1)

    ds = ds.drop(columns=[ 'Precip_Rain'])
    
    ds['Formatted_Date'] = pd.to_datetime(ds['Formatted_Date'], errors='coerce', utc=True)

    # Convert the "Formatted_Date" column to datetime
    ds['Formatted_Date'] = pd.to_datetime(ds['Formatted_Date'], format='%Y-%m-%d %H:%M:%S.%f %z')

    # Extract year, month, day, and hour in seconds
    ds['Year'] = ds['Formatted_Date'].dt.year
    ds['Month'] = ds['Formatted_Date'].dt.month
    ds['Hour'] = ds['Formatted_Date'].dt.hour
    
    ds.loc[ds['Year'] == 2005, 'Year'] = 2006

    scaler = MinMaxScaler()
    ds[columns.columns_to_scale] = scaler.fit_transform(ds[columns.columns_to_scale])
    
    ds = ds.drop(columns.drop_columns, axis=1)
    
    return ds