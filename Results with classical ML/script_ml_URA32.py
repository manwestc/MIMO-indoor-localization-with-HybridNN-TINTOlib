# -*- coding: utf-8 -*-
#!pip install lazypredict
#!pip install tqdm
#!pip install xgboost
#!pip install catboost
#!pip install lightgbm
#!pip install pytest

# Load libraries
#import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, BayesianRidge
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import time
from joblib import Parallel, delayed


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
dataset = pd.read_csv('URA_lab_LoS_32_100_v3.csv')

#df_var = dataset.iloc[:,0:-2]
columns_to_normalize = dataset.columns[:-2]
df_var = (dataset[columns_to_normalize] - dataset[columns_to_normalize].min()) / (dataset[columns_to_normalize].max() - dataset[columns_to_normalize].min())
df_X = dataset.iloc[:,-2]
df_Y = dataset.iloc[:,-1]

np.random.seed(64)
df_var = df_var.sample(frac=1).reset_index(drop=True)

np.random.seed(64)
df_X = df_X.sample(frac=1).reset_index(drop=True)

np.random.seed(64)
df_Y = df_Y.sample(frac=1).reset_index(drop=True)



# Training size
trainings_size = 0.85                     # 85% training set
validation_size = 0.1                     # 10% validation set
test_size = 0.05                         # 5% test set

# Para posicion X
X_train = df_var.iloc[:int(trainings_size*len(df_var))]
y_train_x = df_X.iloc[:int(trainings_size*len(df_X))]
y_train_y = df_Y.iloc[:int(trainings_size*len(df_Y))]

X_val = df_var.iloc[int(trainings_size*len(df_var)):int((trainings_size+validation_size)*len(df_var))]
y_val_x = df_X.iloc[int(trainings_size*len(df_X)):int((trainings_size+validation_size)*len(df_X))]
y_val_y = df_Y.iloc[int(trainings_size*len(df_Y)):int((trainings_size+validation_size)*len(df_Y))]

X_test = df_var.iloc[-int(test_size*len(df_var)):]
y_test_x = df_X.iloc[-int(test_size*len(df_X)):]
y_test_y = df_Y.iloc[-int(test_size*len(df_Y)):]

validation_x = y_val_x
test_x = y_test_x
validation_y = y_val_y
test_y = y_test_y

seed = 7

# Spot Check Algorithms
models = []
models.append(('ADA', AdaBoostRegressor(random_state=seed)))
models.append(('Bagging', BaggingRegressor(random_state=seed)))
models.append(('BayRidge', BayesianRidge()))
models.append(('CART', DecisionTreeRegressor(random_state=seed)))
models.append(('ET', ExtraTreesRegressor(random_state=seed)))
models.append(('GBM', GradientBoostingRegressor(random_state=seed)))
models.append(('HGBR', HistGradientBoostingRegressor(random_state=seed)))
models.append(('k-NN', KNeighborsRegressor(n_neighbors=11)))
models.append(('LiR', LinearRegression()))
models.append(('XGBR', xgb.XGBRegressor(objective="reg:linear", random_state=seed)))
models.append(('RF', RandomForestRegressor(random_state=seed)))

n_jobs = len(models)

def true_dist(y_pred, y_true):
    return np.mean(np.sqrt(
        np.square(np.abs(y_pred[:,0] - y_true[:,0]))
        + np.square(np.abs(y_pred[:,1] - y_true[:,1]))
        ))

def obtain_results_x(name, model):
    #kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    t0 = time.time()
    model.fit(X_train, y_train_x)
    print("----------------------------", model, "---------------------")
    print("##### RESULTS FOR VALIDATION OF X #####")
    predX_val = model.predict(X_val)
    print('Model FPG - R2: %.3f' % r2_score(y_val_x, predX_val))
    print('Model FPG - Root Mean Squared Error: %.3f' % mean_squared_error(y_val_x, predX_val, squared=False)) # RMSE
    print('Model FPG - Mean Absolute Error: %.3f' % mean_absolute_error(y_val_x, predX_val))
    print('Model FPG - Mean Squared Error: %.3f' % mean_squared_error(y_val_x, predX_val)) #MSE
    print("TIME OF VAL: ", time.time()-t0)
    
    print("##### RESULTS FOR TEST OF X #####")
    t0 = time.time()
    predX_test = model.predict(X_test)
    print('Model FPG - R2: %.3f' % r2_score(y_test_x, predX_test))
    print('Model FPG - Root Mean Squared Error: %.3f' % mean_squared_error(y_test_x, predX_test, squared=False)) # RMSE    
    print('Model FPG - Mean Absolute Error: %.3f' % mean_absolute_error(y_test_x, predX_test))
    print('Model FPG - Mean Squared Error: %.3f' % mean_squared_error(y_test_x, predX_test)) #MSE
    print("TIME OF TEST: ", time.time()-t0)

    print("---")
    print("---")
    print("---")

    t0 = time.time()
    model.fit(X_train, y_train_y)
    print("----------------------------", model, "---------------------")
    print("##### RESULTS FOR VALIDATION OF Y #####")
    predY_val = model.predict(X_val)
    print('Model FPG - R2: %.3f' % r2_score(y_val_y, predY_val))
    print('Model FPG - Root Mean Squared Error: %.3f' % mean_squared_error(y_val_y, predY_val, squared=False)) # RMSE
    print('Model FPG - Mean Absolute Error: %.3f' % mean_absolute_error(y_val_y, predY_val))
    print('Model FPG - Mean Squared Error: %.3f' % mean_squared_error(y_val_y, predY_val)) #MSE
    print("TIME OF VAL: ", time.time()-t0)
    
    print("##### RESULTS FOR TEST OF Y #####")
    t0 = time.time()
    predY_test = model.predict(X_test)
    print('Model FPG - R2: %.3f' % r2_score(y_test_y, predY_test))
    print('Model FPG - Root Mean Squared Error: %.3f' % mean_squared_error(y_test_y, predY_test, squared=False)) # RMSE    
    print('Model FPG - Mean Absolute Error: %.3f' % mean_absolute_error(y_test_y, predY_test))
    print('Model FPG - Mean Squared Error: %.3f' % mean_squared_error(y_test_y, predY_test)) #MSE
    print("TIME OF TEST: ", time.time()-t0)
    
    print("---")
    print("---")
    print("---")    

    print("# PREDICTIONS", model ," #")
    predictions_valid = pd.DataFrame()
    predictions_valid["realX"] = validation_x
    predictions_valid["realY"] = validation_y
    
    predictions_valid["predX"] = predX_val
    predictions_valid["predY"] = predY_val
    
    error_valid = true_dist(predictions_valid[["predX", "predY"]].to_numpy(), predictions_valid[["realX", "realY"]].to_numpy())
    print("Validation Error: ", error_valid)
    
    predictions_test = pd.DataFrame()
    predictions_test["realX"] = test_x
    predictions_test["realY"] = test_y
    
    predictions_test["predX"] = predX_test
    predictions_test["predY"] = predY_test
    
    error_test = true_dist(predictions_test[["predX", "predY"]].to_numpy(), predictions_test[["realX", "realY"]].to_numpy())
    print("Test Error: ", error_test)
    print()
    print()


#results  = {name: obtain_results(name, model) for name, model in models}
Parallel(n_jobs = n_jobs)(delayed(obtain_results_x)(name, model) for name, model in models)