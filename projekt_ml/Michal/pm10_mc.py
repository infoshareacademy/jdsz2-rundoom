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
from tpot import TPOTClassifier


# read data to dataframe
df = pd.read_excel('pyly_2.xlsx', sheet_name='Worksheet')

# add columns - hour, day of the week, month
df['Godzina'] = df['Czas mierzenia'].apply(lambda x: int(datetime.datetime.strptime(x, '%m/%d/%Y %H:%M').time().hour))
df['Dzień tygodnia'] = df['Czas mierzenia'].apply(lambda x: int(datetime.datetime.strptime(x, '%m/%d/%Y %H:%M').date().weekday()))
df['Miesiąc'] = df['Czas mierzenia'].apply(lambda x: int(datetime.datetime.strptime(x, '%m/%d/%Y %H:%M').date().month))

# prediction time offset
dni_prognozy = 12

# creating prediction column
df['PM10_P'] = df['PM10'].shift(periods=-dni_prognozy)
df = df.dropna(subset=['PM10_P'])

# identify feature columns and target
X = df[['PM1', 'PM25', 'PM10', 'Temperatura', 'Ciśnienie', 'Prędkość wiatru', 'Wind bearing', 'Godzina', 'Dzień tygodnia', 'Miesiąc']]
y = df['PM10_P']

# split data set into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


def prediction(model):
    model.fit(X_train, y_train.values.ravel())
    # cross validation
    print("____CROSS VALIDATION____")
    cv_score = abs(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error'))
    print("mean cross validation score (mae) is {}".format(cv_score.mean()))
    y_pred = model.predict(X_test)

    # saving predictions to file
    # dataset = pd.DataFrame({'y_pred': y_pred})
    # dataset['y_test'] = y_test
    # dataset = dataset.dropna().reset_index()
    # dataset.to_csv('wyniki.csv', sep=';')

    mae = mean_absolute_error(y_test, y_pred)
    print("{} - Mean Absolute Error:  {}".format(model, mae))

# Prediction for LinearRegression
model_linreg = LinearRegression()
prediction(model_linreg)
# Prediction for RandomForest
model_rf = RandomForestRegressor()
prediction(model_rf)
# Prediction for XGBoost
model_xgb = XGBRegressor()
prediction(model_xgb)





def select_model(X, Y):
    best_models = {}
    models = [
        {
            'name': 'LinearRegression',
            'estimator': LinearRegression(),
            'hyperparameters': {},
        },
        {
            'name': 'KNeighbors',
            'estimator': KNeighborsRegressor(),
            'hyperparameters': {
                'n_neighbors': range(3, 50, 3),
                'weights': ['distance', 'uniform'],
                'algorithm': ['auto'],
                'leaf_size': list(range(10, 51, 10)),
            }
        },

        {
            'name': 'RandomForest',
            'estimator': RandomForestRegressor(),
            'hyperparameters': {
                'bootstrap': ['True'],
                'criterion': ['mae'],
                'max_features': ['auto'],
                'min_samples_leaf': [1, 2, 5],
                'min_samples_split': [2, 4, 6],
                'n_estimators': [10, 20]

            }
        },

        {
            'name': 'XGBoost',
            'estimator': XGBRegressor(),
            'hyperparameters': {
                'gamma': [i / 10.0 for i in range(0, 5)],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': range(100, 200, 10),
                'max_depth': [3, 5, 7, 9],
                'min_child_weight': [1, 3, 5, 6]
            }
        }

    ]

    for model in models:
        print('\n', '-'*20, '\n', model['name'])
        start = time.perf_counter()
        grid = GridSearchCV(model['estimator'], param_grid=model['hyperparameters'], cv=5, scoring="neg_mean_absolute_error",
                            verbose=False, n_jobs=1)
        grid.fit(X, Y.values.ravel())
        best_models[model['name']] = {'score': grid.best_score_, 'params': grid.best_params_}
        run = time.perf_counter() - start
        print('mea: {}\n{} --{:.2f} seconds.'.format(str(abs(grid.best_score_)), str(grid.best_params_), run))

    return best_models
# 'bootstrap': 'True', 'criterion': 'mae', 'max_features': 0.55, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 20

X, y = X_train, y_train
best = select_model(X, y)

# RandomForest
# mea: 9.721994005641747
# {'bootstrap': 'True', 'criterion': 'mae', 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 20} --692.51 seconds.
# RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=None,
#            max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=None,
#            oob_score=False, random_state=None, verbose=0, warm_start=False) - Mean Absolute Error:  8.950282087447109