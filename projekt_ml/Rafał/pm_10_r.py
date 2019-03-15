import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



pd.set_option("display.max_columns", 30)

# read data to dataframe
df = pd.read_excel('pyly_2.xlsx', sheet_name='Worksheet')

# add columns - hour, day of the week, month
df['Godzina'] = df['Czas mierzenia'].apply(lambda x: int(datetime.datetime.strptime(x, '%m/%d/%Y %H:%M').time().hour))
df['Dzień tygodnia'] = df['Czas mierzenia'].apply(lambda x: int(datetime.datetime.strptime(x, '%m/%d/%Y %H:%M').date().weekday()))
df['Miesiąc'] = df['Czas mierzenia'].apply(lambda x: int(datetime.datetime.strptime(x, '%m/%d/%Y %H:%M').date().month))
df['Czas mierzenia'] = df['Czas mierzenia'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M'))

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
krok = 72

por_sk = y[krok-1:]

pred_sk = movingavg(sk,krok)

mse = sum(abs(pred_sk - por_sk))/len(por_sk)

print(mse)

czas_x =df['Czas mierzenia'][krok-1+3:]




plt.figure(figsize=(10, 10))
plt.plot(czas_x[-100:].values,por_sk[-100:], label = 'real')
plt.plot(czas_x[-100:].values,pred_sk[-100:], label = 'pred')
plt.xticks(rotation = 40)
plt.legend()
plt.show()


