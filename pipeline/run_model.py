from train import train_model
from test import test_model
import pandas as pd
from preprocessing import preprocess_train_data
from preprocessing import preprocess_testing_data


train_model(file_name="train.csv",model_name='RandomForestRegressor')
test_model(file_name='new_input.csv',model_name='RandomForestRegressor') 