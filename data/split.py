import pandas as pd
from sklearn.model_selection import train_test_split

ds = pd.read_csv('D:/3Kurs/1Sem/SS/Practice/rgr/data/weatherHistory_fixed.csv')
# print(ds)

train, test = train_test_split(ds, test_size=0.1, random_state=42)

train.to_csv('D:/3Kurs/1Sem/SS/Practice/rgr/data/train.csv', index=False)
test.to_csv('D:/3Kurs/1Sem/SS/Practice/rgr/data/new_input.csv', index=False)