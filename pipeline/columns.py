outliers_left_columns = ['Temperature','Apparent_Temperature','Humidity','Pressure']

outliers_right_columns = [ 'Wind_Speed', 'Pressure']

no_need_columns = ['Apparent_Temperature', 'Loud_Cover', 'Daily_Summary']

columns_to_scale = ['Temperature','Wind_Bearing', 'Pressure', 'Wind_Speed']

drop_columns = ['Formatted_Date', 'Summary', 'Precip_Type','Unnamed: 0']

X_column = ['Temperature', 'Wind_Speed', 'Wind_Bearing', 'Visibility',
       'Pressure', 'Partly Cloudy', 'Humid', 'Rain', 'Dry', 'Breezy',
       'Light Rain', 'Clear', 'Overcast', 'Windy', 'Dangerously Windy',
       'Foggy', 'Drizzle', 'Mostly Cloudy', 'Snow', 'Year', 'Month', 'Hour']

y_column = ['Humidity']