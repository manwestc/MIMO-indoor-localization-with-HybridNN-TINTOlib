# For example, this script execute the ULA-64 antennas dataset in slurm for each algorithm. 
# Executed in slurm with the following script:
"""
#!/bin/bash
#SBATCH --job-name=ml
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150GB
#SBATCH --array=0-8
#SBATCH --output=ml/slurm-%A_%a.out
#SBATCH --error=ml/slurm-%A_%a.err

cd $SLURM_SUBMIT_DIR

micromamba activate env
DATASET_PATH="/data/datasets/ULA_lab_LoS_64.csv" # Change the dataset to the one you want to use (ULA, URA, DIS setup; 8, 16, 32, 64 antennas)


# Execute with: sbatch ml.sh
python script_ml.py $SLURM_ARRAY_TASK_ID $DATASET_PATH
"""

import numpy as np
import pandas as pd
import time
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    DecisionTreeRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

dataset = pd.read_csv(sys.argv[2])

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

# Get the algorithm to execute
index = sys.argv[1]
index = int(index)

# Spot Check Algorithms
models = []
models.append(('ADA', AdaBoostRegressor(random_state=seed)))
models.append(('Bagging', BaggingRegressor(random_state=seed)))
models.append(('CART', DecisionTreeRegressor(random_state=seed)))
models.append(('ET', ExtraTreesRegressor(random_state=seed)))
models.append(('GBM', GradientBoostingRegressor(random_state=seed)))
models.append(('HGBR', HistGradientBoostingRegressor(random_state=seed)))
models.append(('k-NN', KNeighborsRegressor(n_neighbors=11)))
models.append(('LiR', LinearRegression()))
models.append(('XGBR', xgb.XGBRegressor(objective="reg:linear", random_state=seed)))


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
    print("##### X-VALIDATION RESULTS #####")
    predX_val = model.predict(X_val)
    print('R2: %.3f' % r2_score(y_val_x, predX_val))
    print('Root Mean Squared Error: %.3f' % mean_squared_error(y_val_x, predX_val, squared=False)) # RMSE
    print('Mean Absolute Error: %.3f' % mean_absolute_error(y_val_x, predX_val))
    print('Mean Squared Error: %.3f' % mean_squared_error(y_val_x, predX_val)) #MSE
    print("VALIDATION TIME: ", time.time()-t0)
    
    print("##### X-TEST RESULTS #####")
    t0 = time.time()
    predX_test = model.predict(X_test)
    print('R2: %.3f' % r2_score(y_test_x, predX_test))
    print('Root Mean Squared Error: %.3f' % mean_squared_error(y_test_x, predX_test, squared=False)) # RMSE    
    print('Mean Absolute Error: %.3f' % mean_absolute_error(y_test_x, predX_test))
    print('Mean Squared Error: %.3f' % mean_squared_error(y_test_x, predX_test)) #MSE
    print("TEST TIME: ", time.time()-t0)

    print("---")
    print("---")
    print("---")

    t0 = time.time()
    model.fit(X_train, y_train_y)
    print("----------------------------", model, "---------------------")
    print("##### Y-VALIDATION RESULTS #####")
    predY_val = model.predict(X_val)
    print('R2: %.3f' % r2_score(y_val_y, predY_val))
    print('Root Mean Squared Error: %.3f' % mean_squared_error(y_val_y, predY_val, squared=False)) # RMSE
    print('Mean Absolute Error: %.3f' % mean_absolute_error(y_val_y, predY_val))
    print('Mean Squared Error: %.3f' % mean_squared_error(y_val_y, predY_val)) #MSE
    print("VALIDATION TIME: ", time.time()-t0)
    
    print("##### Y-TEST RESULTS #####")
    t0 = time.time()
    predY_test = model.predict(X_test)
    print('R2: %.3f' % r2_score(y_test_y, predY_test))
    print('Root Mean Squared Error: %.3f' % mean_squared_error(y_test_y, predY_test, squared=False)) # RMSE    
    print('Mean Absolute Error: %.3f' % mean_absolute_error(y_test_y, predY_test))
    print('Mean Squared Error: %.3f' % mean_squared_error(y_test_y, predY_test)) #MSE
    print("TEST TIME: ", time.time()-t0)
    
    print("---")
    print("---")
    print("---")    

    print("# PREDICTIONS", model ," #")
    validation_preds = pd.DataFrame()
    validation_preds["realX"] = validation_x
    validation_preds["realY"] = validation_y
    
    validation_preds["predX"] = predX_val
    validation_preds["predY"] = predY_val
    
    val_error = true_dist(validation_preds[["predX", "predY"]].to_numpy(), validation_preds[["realX", "realY"]].to_numpy())
    print("vALIDATION MEAN ERROR: ", val_error)
    
    predicciones_test = pd.DataFrame()
    predicciones_test["realX"] = test_x
    predicciones_test["realY"] = test_y
    
    predicciones_test["predX"] = predX_test
    predicciones_test["predY"] = predY_test
    
    error_test = true_dist(predicciones_test[["predX", "predY"]].to_numpy(), predicciones_test[["realX", "realY"]].to_numpy())
    print("TEST MEAN ERROR: ", error_test)
    print()
    print()

# Execute the corresponding algorithm
name, model = models[index]
obtain_results_x(name, model)