import pandas as pd
import pickle 
from preprocessing import preprocess_train_data
import columns
from sklearn.ensemble import RandomForestRegressor
import hiperparameters

def train_model(file_name: str = 'train.csv', model_name: str = 'RandomForestRegressor'):
    # loading data
    ds = pd.read_csv('D:/3Kurs/1Sem/SS/Practice/rgr/data/' + file_name)
    
    ds = preprocess_train_data(ds)
    
    X = ds[columns.X_column]
    y = ds[columns.y_column]
    
    rf_model = RandomForestRegressor(**hiperparameters.param_grid)
    rf_model.fit(X, y)
    
    with open(f'D:/3Kurs/1Sem/SS/Practice/rgr/models/RandomForestRegressor.pickle', 'wb') as handle:
        pickle.dump(rf_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
