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
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import sys


pd.set_option("display.max_columns", 30)

# read data to dataframe
df = pd.read_excel('pyly_2.xlsx', sheet_name='Worksheet')

# add columns - hour, day of the week, month
df['Godzina'] = df['Czas mierzenia'].apply(lambda x: int(datetime.datetime.strptime(x, '%m/%d/%Y %H:%M').time().hour))
df['Dzień tygodnia'] = df['Czas mierzenia'].apply(lambda x: int(datetime.datetime.strptime(x, '%m/%d/%Y %H:%M').date().weekday()))
df['Miesiąc'] = df['Czas mierzenia'].apply(lambda x: int(datetime.datetime.strptime(x, '%m/%d/%Y %H:%M').date().month))

# prediction time offset
godziny_prognozy = 12


# creating prediction column
df['PM10_P'] = df['PM10'].shift(periods=-godziny_prognozy)
df = df.dropna(subset=['PM10_P'])


# identify feature columns and target
X = df[['PM1', 'PM25', 'PM10', 'Temperatura', 'Ciśnienie', 'Prędkość wiatru', 'Wind bearing', 'Godzina', 'Dzień tygodnia', 'Miesiąc']]
y = df['PM10_P']

X = X.iloc[3:]
X = X.reset_index(drop=True)
y = y[3:]





#Średnia Krocząca

def movingavg(values, window):
    weights = np.repeat(1.0,window)/window
    smas = np.convolve(values,weights,'valid')
    return smas


sk = y
krok = 120

por_sk = y[krok-1:]

pred_sk = movingavg(sk,krok)

mse = sum(abs(pred_sk - por_sk))/len(por_sk)

print(mse)

czas_x =df['Czas mierzenia'][krok-1+3:]




plt.figure(figsize=(10, 10))
plt.plot(czas_x[-100:].values,por_sk[-100:], label = 'real')
plt.plot(czas_x[-100:].values,pred_sk[-100:], label = 'pred')
plt.xticks(rotation = 40)
ax1 = plt.axes()
ax1.axes.get_xaxis().set_visible(False)
plt.legend()
plt.show()







#df1 = df['Czas mierzenia']

# categorical_columns = ["Godzina", "Dzień tygodnia", "Miesiąc"]
#
# #Zamieniamy feature'y ketegoryczne na reprezentacje liczbową
# X = pd.get_dummies(X, columns=categorical_columns)


#simple_model['PM10_P'] = y

#simple_model.to_csv("simple.csv")


# for k in range(len(simple_model)):
#     if (simple_model['PM10'][k] > 50):
#         PM.append('do_50')
#     else:
#         PM.append('pow_50')
#
# simple_model['PM'] = PM

# print(simple_model.groupby(['PM']).mean())


#print( np.mean(simple_model.loc[ (simple_model['Temperatura']>=-11)  & (simple_model['Temperatura']<-0.5) ]['PM10_P'] ))

