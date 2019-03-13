import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import datetime
import time
from sklearn.model_selection import cross_val_score
#from tpot import TPOTClassifier
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
X = df[['Temperatura', 'Ciśnienie', 'Prędkość wiatru', 'Wind bearing', 'Godzina', 'Dzień tygodnia', 'Miesiąc', 'Wilgotność']]
y = df['PM10_P']

X = X.iloc[3:]
X = X.reset_index(drop=True)
y = y[3:]


# maskZ = ((X.Miesiąc >= 10) | (X.Miesiąc <= 3))
#
# maskL = ((X.Miesiąc >= 4) & (X.Miesiąc <= 9))
#
# column_name = 'Pora Roku'
# X.loc[maskZ, column_name] = -1
# X.loc[maskL, column_name] = 1



#X['GPR'] = X['Pora Roku']*X['Godzina']
#X['PKW'] = X['Prędkość wiatru']*X['Wind bearing']
# X['TW'] = X['Temperatura']*X['Wilgotność']
# X['CW'] = X['Ciśnienie']*X['Wilgotność']


X['TC'] = X['Temperatura']*X['Ciśnienie']
columns_to_delete = ['Wilgotność','Temperatura']
X = X.drop(columns=columns_to_delete, axis=1)



# split data set into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


#'PM1', 'PM25','PM10',

# RF = {
#                 'max_d_range': list(range(10,30,10)),
#                 'min_samples_leaf': [2, 3],
#                 'min_samples_split': [2, 4],
#                 'n_estimators': list(range(50,200,30))
# }
#
#
# param_grid = dict(max_depth = RF['max_d_range'],min_samples_leaf= RF['min_samples_leaf'],criterion = ['mae'], bootstrap = ['True'],
#                   n_estimators=RF['n_estimators'],min_samples_split = RF['min_samples_split'] )



regresor = RandomForestRegressor(random_state=101)


# GRIDCV
#{'bootstrap': 'True', 'criterion': 'mae', 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
#,max_depth= 20, min_samples_leaf= 2, bootstrap=True,min_samples_split= 2, n_estimators= 100, criterion= 'mae'

# grid = GridSearchCV(regresor, param_grid = param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs= -1, verbose=False)
#
# grid.fit(X_train,y_train.values.ravel())
#
# print(grid.best_params_)


regresor.fit(X_train, y_train.values.ravel())

cv_score = abs(cross_val_score(regresor, X_train, y_train, cv=5, scoring='neg_mean_absolute_error'))
print("mean cross validation score (mae) is {}".format(cv_score.mean()))
y_pred = regresor.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
print("{} - Mean Absolute Error:  {}".format(regresor, mae))


print(regresor.feature_importances_)


