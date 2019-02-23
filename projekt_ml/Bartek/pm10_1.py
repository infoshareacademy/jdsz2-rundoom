import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import datetime
import sys

pd.set_option("display.max_columns", 30)

# Load spreadsheet
xl = pd.ExcelFile('pyly_2.xlsx')

# Print the sheet names
print(xl.sheet_names)

# Load a sheet into a DataFrame by name: df1
df1 = xl.parse('Worksheet')

# print(df1.head())
# print(df1.info())

df1['Czas'] = df1['Czas mierzenia'].apply(lambda x: int(datetime.datetime.strptime(x, '%m/%d/%Y %H:%M').time().hour))
df1['Dzień tygodnia'] = df1['Czas mierzenia'].apply(lambda x: int(datetime.datetime.strptime(x, '%m/%d/%Y %H:%M').date().weekday()))
df1['Miesiąc'] = df1['Czas mierzenia'].apply(lambda x: int(datetime.datetime.strptime(x, '%m/%d/%Y %H:%M').date().month))

df1['Czas mierzenia'] = df1['Czas mierzenia'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M'))

df2 = df1['Czas mierzenia']

# print(df1.head())
# print(df1.shape)

dni_prognozy = 12

for i, row in df1.iterrows():
    if i >= dni_prognozy:
        df1.loc[i - dni_prognozy, 'PM10_P'] = row['PM10']

y = df1.loc[3:df1.shape[0]-dni_prognozy-1, 'PM10_P']
# print(y.shape)

# Lista kolumn które chcemy usunąć.
columns_to_delete = ["ID czujnika", "AQI", "O3", "CO", "SO2", "NO2", "C6H6", "CH2O", "Szerokość geo.", "Wysokość geo.", "Epoch", "is_forecast", "Czas mierzenia", "PM10_P"]

# columns_to_delete = ["ID czujnika", "AQI", "PM1","PM25" ,"PM10" ,"O3", "CO", "SO2", "NO2", "C6H6", "CH2O", "Szerokość geo.", "Wysokość geo.", "Epoch", "is_forecast", "Czas mierzenia", "PM10_P"]

# Lista kolumn z wartościami które poddamy "dummifikacji"
categorical_columns = ["Czas", "Dzień tygodnia", "Miesiąc"]

# Usuwamy kolumny wcześniej zdefiniowane w liście columns_to_delete
df1 = df1.drop(columns=columns_to_delete, axis=1)

# Zamieniamy feature'y ketegoryczne na reprezentacje liczbową
# df1 = pd.get_dummies(df1, columns=categorical_columns)

# sys.exit(0)

X = df1.loc[3:df1.shape[0]-dni_prognozy-1]

X_prog = df1[df1.shape[0]-dni_prognozy:]

# print(X.head())
# print(X.shape)
# print(X.describe())

# sys.exit(0)

# print(data.tail(10))
# print(target.tail(10))

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=101)

model = RandomForestRegressor(n_estimators=25, random_state=0)

#print(cross_val_score(model, train_X, train_y, cv=5).mean())

model.fit(train_X, train_y)

pred_test_y = model.predict(test_X)
pred_y = model.predict(X)
prog_y = model.predict(X_prog)

mae = mean_absolute_error(test_y, pred_test_y)

print("Mean Absolute Error:  {}".format(mae))

plt.figure(figsize=(10, 10))
plt.title('Dopasowanie modelu')
plt.plot(df2[-50-dni_prognozy:-dni_prognozy].values, y[-50:], label='real')
plt.plot(df2[-50-dni_prognozy:-dni_prognozy].values, pred_y[-50:], label='pred')
# plt.plot(np.arange(50, 51+dni_prognozy, 1), np.concatenate([pred_y[-1].ravel(), prog_y.ravel()]), label='prog')
plt.xlabel('Data i godzina')
plt.ylabel('PM 10')
plt.legend()
plt.show()

plt.figure(figsize=(10, 10))
plt.title('Prognoza')
plt.plot(df2[-50-dni_prognozy:-dni_prognozy].values, y[-50:], label='real')
# plt.plot(np.arange(1, 51, 1), pred_y[-50:], label='pred')
plt.plot(df2[-1-dni_prognozy:].values, np.concatenate([y[-1:], prog_y.ravel()]), label='prog')
plt.xlabel('Data i godzina')
plt.ylabel('PM 10')
plt.legend()
plt.show()
