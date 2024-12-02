# For example, this script execute the HyNN in ULA-64 antennas dataset, in slurm. 
# Executed in slurm with the following script:
"""
#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=300GB
#SBATCH --output=ablation/slurm-%A_%a.out
#SBATCH --error=ablation/slurm-%A_%a.err

cd $SLURM_SUBMIT_DIR

source $HOME/.bashrc
micromamba activate gpu_env

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

DATASET_PATH="/data/datasets/ULA_lab_LoS_64.csv" # Change the dataset to the one you want to use (ULA, URA, DIS setup; 8, 16, 32, 64 antennas)
SETUP = "ULA" # Change the setup to the one you want to use (ULA, URA, DIS)
ANTENNAS = "64" # Change the number of antennas to the one you want to use (8, 16, 32, 64)

# Execute with: sbatch ablation.sh
python ablation.py $DATASET_PATH $SETUP $ANTENNAS
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import time
import gc

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Multiply, Add, Concatenate, Dense, Conv2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Activation,MaxPooling2D
from keras.layers import MultiHeadAttention, LayerNormalization, Lambda

from TINTOlib.tinto import TINTO

df = pd.read_csv(sys.argv[1])
setup = sys.argv[2] # In string format
num_antennas = sys.argv[3] # In string format

#Select the model and the parameters
problem_type = "regression"
pixel = 35
image_model = TINTO(problem= problem_type,pixels=pixel,blur=True)

images_folder = "/images_" + num_antennas + "antenas_" + setup # "/images_64antenas_ULA"
results_folder = "/results_ablation/"



# * NORMALIZE DATASET

# Select all the attributes to normalize
columns_to_normalize = df.columns[:-2]

# Normalize between 0 and 1
df_normalized = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())

# Combine the attributes and the label
#df_normalized = pd.concat([df_normalized, df[df.columns[-1]]], axis=1)
df_normalized = pd.concat([df_normalized, df[df.columns[-2]], df[df.columns[-1]]], axis=1)


#Generate the images with TINTO

if not os.path.exists(images_folder):
   print("generating images...")
   image_model.generateImages(df.iloc[:,:-1], images_folder)
   print("Images generated")
   
   # Save the TINTO model
   pickle.dump(image_model, open(images_folder + "/image_model.pkl", "wb"))
   

if not os.path.exists(images_folder+results_folder):
   os.makedirs(images_folder+results_folder)
   
img_paths = os.path.join(images_folder,problem_type+".csv")


imgs = pd.read_csv(img_paths) 

imgs["images"]= images_folder + "/" + imgs["images"] 

imgs["images"] = imgs["images"].str.replace("\\","/")


combined_dataset_x = pd.concat([imgs,df_normalized.iloc[:,:-1]],axis=1)
combined_dataset_y = pd.concat([imgs,pd.concat([df_normalized.iloc[:,:-2], df_normalized.iloc[:,-1:]],axis=1)],axis=1)  

#df_x = combined_dataset.drop("homa_b",axis=1).drop("values",axis=1)
df_x = combined_dataset_x.drop("PositionX",axis=1).drop("values",axis=1)
df_y_for_x = combined_dataset_x["values"]
df_y_for_y = combined_dataset_y["PositionY"]

np.random.seed(64)
df_x = df_x.sample(frac=1).reset_index(drop=True)

np.random.seed(64)
df_y_for_x = df_y_for_x.sample(frac=1).reset_index(drop=True)

np.random.seed(64)
df_y_for_y = df_y_for_y.sample(frac=1).reset_index(drop=True)


# Training size
trainings_size = 0.85                     # 85% training set
validation_size = 0.1                     # 10% validation set
test_size = 0.05                         # 5% test set

import cv2

# Split the dataset into training, validation and test sets
X_train = df_x.iloc[:int(trainings_size*len(df_x))]
y_train_x = df_y_for_x.iloc[:int(trainings_size*len(df_y_for_x))]
y_train_y = df_y_for_y.iloc[:int(trainings_size*len(df_y_for_y))]

X_val = df_x.iloc[int(trainings_size*len(df_x)):int((trainings_size+validation_size)*len(df_x))]
y_val_x = df_y_for_x.iloc[int(trainings_size*len(df_y_for_x)):int((trainings_size+validation_size)*len(df_y_for_x))]
y_val_y = df_y_for_y.iloc[int(trainings_size*len(df_y_for_y)):int((trainings_size+validation_size)*len(df_y_for_y))]

X_test = df_x.iloc[-int(test_size*len(df_x)):]
y_test_x = df_y_for_x.iloc[-int(test_size*len(df_y_for_x)):]
y_test_y = df_y_for_y.iloc[-int(test_size*len(df_y_for_y)):]

X_train_num = X_train.drop("images",axis=1)
X_val_num = X_val.drop("images",axis=1)
X_test_num = X_test.drop("images",axis=1)

# For 3 RGB channels
X_train_img = np.array([cv2.resize(cv2.imread(img),(pixel,pixel)) for img in X_train["images"]])
X_val_img = np.array([cv2.resize(cv2.imread(img),(pixel,pixel)) for img in X_val["images"]])
X_test_img = np.array([cv2.resize(cv2.imread(img),(pixel,pixel)) for img in X_test["images"]])
   

validation_x = y_val_x
test_x = y_test_x
validation_y = y_val_y
test_y = y_test_y

shape = len(X_train_num.columns)


# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
#   *                          MLP MODEL FOR X
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

from keras.layers import AveragePooling2D, Concatenate

tf.keras.backend.clear_session()
gc.collect()

dropout = 0.3

filters_ffnn = [1024,512,256,128,64,32,16]

ff_inputs = Input(shape = (shape,))

# * START BRANCH 1
mlp_1 = Dense(1024, activation='relu')(ff_inputs)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(512, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(256, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(128, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(64, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(32, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

mlp_1 = Dense(16, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)

# * START BRANCH 2
mlp_2 = Dense(1024, activation='relu')(ff_inputs)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(512, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(256, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(128, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(64, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(32, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

mlp_2 = Dense(16, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)

merged_tabular = Concatenate(axis=1)([mlp_1, mlp_2])

x = Dense(64, activation="relu")(merged_tabular)
x = Dense(32, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dense(1, activation="linear")(x)

ff_modelX = Model(inputs = ff_inputs, outputs = x)

from tensorflow_addons.metrics import RSquare


METRICS = [
   tf.keras.metrics.MeanSquaredError(name = 'mse'),
   tf.keras.metrics.MeanAbsoluteError(name = 'mae'),
   #tf.keras.metrics.R2Score(name = 'r2'),
   RSquare(name='r2_score'),
   tf.keras.metrics.RootMeanSquaredError(name = 'rmse')
]

opt = Adam()
ff_modelX.compile(
   loss="mse",
   optimizer=opt,
   metrics = METRICS
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

t0 = time.time()

model_history=ff_modelX.fit(
   x=X_train_num, y=y_train_x,
   validation_data=(X_val_num, y_val_x),
   epochs=200,
   batch_size=32,
   verbose=2,
   callbacks=[early_stopping]
)
print("TRAIN TIME: ", time.time()-t0)

ff_modelX.save(images_folder+results_folder+'/ff_modelX.h5')
tf.keras.backend.clear_session()
gc.collect()

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
#   *                          MLP MODEL FOR Y
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////

dropout = 0.3


filters_ffnn = [1024,512,256,128,64,32,16]


ff_inputs = Input(shape = (shape,))


from keras.layers import AveragePooling2D, Concatenate


# * START BRANCH 1
mlp_1 = Dense(1024, activation='relu')(ff_inputs)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)


mlp_1 = Dense(512, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)


mlp_1 = Dense(256, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)


mlp_1 = Dense(128, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)


mlp_1 = Dense(64, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)


mlp_1 = Dense(32, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)


mlp_1 = Dense(16, activation='relu')(mlp_1)
mlp_1 = BatchNormalization()(mlp_1)
mlp_1 = Dropout(dropout)(mlp_1)


# * START BRANCH 2
mlp_2 = Dense(1024, activation='relu')(ff_inputs)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)


mlp_2 = Dense(512, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)


mlp_2 = Dense(256, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)


mlp_2 = Dense(128, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)


mlp_2 = Dense(64, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)


mlp_2 = Dense(32, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)


mlp_2 = Dense(16, activation='relu')(mlp_2)
mlp_2 = BatchNormalization()(mlp_2)
mlp_2 = Dropout(dropout)(mlp_2)


merged_tabular = Concatenate(axis=1)([mlp_1, mlp_2])

x = Dense(64, activation="relu")(merged_tabular)
x = Dense(32, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dense(1, activation="linear")(x)

ff_modelY = Model(inputs = ff_inputs, outputs = x)


from tensorflow_addons.metrics import RSquare


METRICS = [
   tf.keras.metrics.MeanSquaredError(name = 'mse'),
   tf.keras.metrics.MeanAbsoluteError(name = 'mae'),
   #tf.keras.metrics.R2Score(name='r2_score'),
   RSquare(name='r2_score'),
   tf.keras.metrics.RootMeanSquaredError(name='rmse')
]


opt = Adam(learning_rate=1e-4)
ff_modelY.compile(
   loss="mse",
   optimizer=opt,
   metrics = METRICS
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

t0 = time.time()

model_history=ff_modelY.fit(
   x=X_train_num, y=y_train_y,
   validation_data=(X_val_num, y_val_y),
   epochs=200,
   batch_size=32,
   verbose=2,
   callbacks=[early_stopping]
   #verbose=2
   #steps_per_epoch = X_train_num.shape[0]//batch_size,
   #validation_steps = X_train_num.shape[0]//batch_size,
)
print("TRAIN TIME: ", time.time()-t0)


ff_modelY.save(images_folder+results_folder+'/ff_modelY.h5')
tf.keras.backend.clear_session()
gc.collect()


def true_dist(y_pred, y_true):
   return np.mean(np.sqrt(
       np.square(np.abs(y_pred[:,0] - y_true[:,0]))
       + np.square(np.abs(y_pred[:,1] - y_true[:,1]))
       ))

# VALIDATION RESULTS
folder = images_folder+results_folder+"/predictions/mlp/validation/"
if not os.path.exists(folder):
   os.makedirs(folder)

start_time = time.time()
predX_val = ff_modelX.predict(X_val_num)
print("PREDICTION TIME OF X (VALIDATION): ", time.time()-start_time)


Start_time = time.time()
predY_val = ff_modelY.predict(X_val_num)
print("PREDICTION TIME OF Y (VALIDATION): ", time.time()-start_time)

predictions_valid = pd.DataFrame()
predictions_valid["realX"] = validation_x 
predictions_valid["realY"] = validation_y 

predictions_valid["predX"] = predX_val
predictions_valid["predY"] = predY_val

predictions_valid.to_csv(folder+'preds_val.csv', index=False)

error_valid = true_dist(predictions_valid[["predX", "predY"]].to_numpy(), predictions_valid[["realX", "realY"]].to_numpy())
print(error_valid)


# RESULTS FOR TEST


folder = images_folder+results_folder+"/predictions/mlp/test/"
if not os.path.exists(folder):
   os.makedirs(folder)

start_time = time.time()
predictionX = ff_modelX.predict(X_test_num)
print("PREDICTION TIME OF X (TEST): ", time.time()-start_time)


start_time = time.time()
predictionY = ff_modelY.predict(X_test_num)
print("PREDICTION TIME OF Y (TEST): ", time.time()-start_time)

predictions_test = pd.DataFrame()
predictions_test["realX"] = test_x
predictions_test["realY"] = test_y

predictions_test["predX"] = predictionX
predictions_test["predY"] = predictionY

predictions_test.to_csv(folder+'preds_test.csv', index=False)

error_test = true_dist(predictions_test[["predX", "predY"]].to_numpy(), predictions_test[["realX", "realY"]].to_numpy())

print(error_test)

mae_x = mean_absolute_error(y_test_x, predictionX)
mse_x = mean_squared_error(y_test_x, predictionX)
rmse_x = mean_squared_error(y_test_x, predictionX, squared=False)
r2_x = r2_score(y_test_x, predictionX)


mae_y = mean_absolute_error(y_test_y, predictionY)
mse_y = mean_squared_error(y_test_y, predictionY)
rmse_y = mean_squared_error(y_test_y, predictionY, squared=False)
r2_y = r2_score(y_test_y, predictionY)

print("TEST X:")
print("Mean Absolute Error:", mae_x)
print("Mean Squared Error:", mse_x)
print("Root Mean Squared Error:", rmse_x)
print("R2 Score:", r2_x)
print()
print("TEST Y:")    
print("Mean Absolute Error:", mae_y)
print("Mean Squared Error:", mse_y)
print("Root Mean Squared Error:", rmse_y)
print("R2 Score:", r2_y)

# Save evaluation metrics to a text file
results_filename = images_folder+results_folder+'evaluation_results_mlp.txt'
with open(results_filename, 'w') as results_file:
   results_file.write("Evaluation Metrics FOR X:\n")
   #results_file.write(f"Mean Absolute Percentage Error: {mape}\n")
   results_file.write(f"Mean Absolute Error: {mae_x}\n")
   results_file.write(f"Mean Squared Error: {mse_x}\n")
   results_file.write(f"Root Mean Squared Error: {rmse_x}\n")
   results_file.write(f"R2 Score: {r2_x}\n")
   results_file.write("\n")
   results_file.write("Evaluation Metrics FOR Y:\n")
   #results_file.write(f"Mean Absolute Percentage Error: {mape}\n")
   results_file.write(f"Mean Absolute Error: {mae_y}\n")
   results_file.write(f"Mean Squared Error: {mse_y}\n")
   results_file.write(f"Root Mean Squared Error: {rmse_y}\n")
   results_file.write(f"R2 Score: {r2_y}\n")
   results_file.write("\n")
   results_file.write(f"Error medio de validacion: {error_valid}\n")
   results_file.write(f"Error medio de test: {error_test}\n")

###################
#### CNN MODEL ####
###################

#Start branch 1
tower_1 = Conv2D(16, (3,3), activation='relu',padding="same")(input_shape)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)


tower_1 = Conv2D(32, (3,3), activation='relu',padding="same")(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)


tower_1 = Conv2D(64, (3,3), activation='relu',padding="same")(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)


tower_1 = Conv2D(64, (3,3), activation='relu',padding="same")(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)
#End branch 1


#Start branch 2
tower_2 = Conv2D(16, (5,5), activation='relu',padding="same")(input_shape)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)


tower_2 = Conv2D(32, (5,5), activation='relu',padding="same")(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)


tower_2 = Conv2D(64, (5,5), activation='relu',padding="same")(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)


tower_2 = Conv2D(64, (5,5), activation='relu',padding="same")(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)
#End branch 2


#Concatenation of the 2 branches
merged = Concatenate(axis=1)([tower_1, tower_2])

#Flattening
merged = Flatten()(merged)


#Additional layers
out = Dense(512, activation='relu')(merged)
out = Dropout(dropout)(merged)
out = Dense(256, activation='sigmoid')(out)
out = Dropout(dropout)(out)
out = Dense(128, activation='sigmoid')(out)
out = Dropout(dropout)(out)
out = Dense(64, activation='sigmoid')(out)
out = Dropout(dropout)(out)
out = Dense(32, activation='sigmoid')(out)
out = Dropout(dropout)(out)

x = Dense(64, activation="relu")(out)
x = Dense(32, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dense(1, activation="linear")(x)

cnn_modelX = Model(inputs = input_shape, outputs = x)

from tensorflow_addons.metrics import RSquare


METRICS = [
   tf.keras.metrics.MeanSquaredError(name = 'mse'),
   tf.keras.metrics.MeanAbsoluteError(name = 'mae'),
   #tf.keras.metrics.R2Score(name = 'r2'),
   RSquare(name='r2_score'),
   tf.keras.metrics.RootMeanSquaredError(name = 'rmse')
]

METRICS = [
   tf.keras.metrics.MeanSquaredError(name = 'mse'),
   tf.keras.metrics.MeanAbsoluteError(name = 'mae'),
   #tf.keras.metrics.R2Score(name = 'r2'),
   RSquare(name='r2_score'),
   tf.keras.metrics.RootMeanSquaredError(name = 'rmse')
]

opt = Adam()
cnn_modelX.compile(
   loss="mse",
   optimizer=opt,
   metrics = METRICS
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


t0 = time.time()

model_history=cnn_modelX.fit(
   x=X_train_img, y=y_train_x,
   validation_data=(X_val_img, y_val_x),
   epochs=200,
   batch_size=32,
   verbose=2,
   callbacks=[early_stopping]
)

print("TRAIN TIME: ", time.time()-t0)


cnn_modelX.save(images_folder+results_folder+'/cnn_modelX.h5')
tf.keras.backend.clear_session()
gc.collect()

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
#   *                           MODEL FOR Y
# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////




#Input
input_shape = Input(shape=(pixel, pixel, 3))


#Start branch 1
tower_1 = Conv2D(16, (3,3), activation='relu',padding="same")(input_shape)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)


tower_1 = Conv2D(32, (3,3), activation='relu',padding="same")(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)


tower_1 = Conv2D(64, (3,3), activation='relu',padding="same")(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)


tower_1 = Conv2D(64, (3,3), activation='relu',padding="same")(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_1 = MaxPooling2D(2,2)(tower_1)
tower_1 = Dropout(dropout)(tower_1)
#End branch 1


#Start branch 2
tower_2 = Conv2D(16, (5,5), activation='relu',padding="same")(input_shape)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)


tower_2 = Conv2D(32, (5,5), activation='relu',padding="same")(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)


tower_2 = Conv2D(64, (5,5), activation='relu',padding="same")(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)


tower_2 = Conv2D(64, (5,5), activation='relu',padding="same")(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Activation('relu')(tower_2)
tower_2 = AveragePooling2D(2,2)(tower_2)
tower_2 = Dropout(dropout)(tower_2)
#End branch 2


#Concatenation of the 2 branches
merged = Concatenate(axis=1)([tower_1, tower_2])


#Flattening
merged = Flatten()(merged)


#Additional layers
out = Dense(512, activation='relu')(merged)
out = Dropout(dropout)(merged)
out = Dense(256, activation='sigmoid')(out)
out = Dropout(dropout)(out)
out = Dense(128, activation='sigmoid')(out)
out = Dropout(dropout)(out)
out = Dense(64, activation='sigmoid')(out)
out = Dropout(dropout)(out)
out = Dense(32, activation='sigmoid')(out)
out = Dropout(dropout)(out)

x = Dense(64, activation="relu")(out)
x = Dense(32, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dense(1, activation="linear")(x)

cnn_modelY = Model(inputs = input_shape, outputs = x)


METRICS = [
   tf.keras.metrics.MeanSquaredError(name = 'mse'),
   tf.keras.metrics.MeanAbsoluteError(name = 'mae'),
   #tf.keras.metrics.R2Score(name='r2_score'),
   RSquare(name='r2_score'),
   tf.keras.metrics.RootMeanSquaredError(name='rmse')
]


opt = Adam(learning_rate=1e-4)
#opt = Adam()
cnn_modelY.compile(
   loss="mse",
   optimizer=opt,
   metrics = METRICS
)


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


t0 = time.time()


model_history=cnn_modelY.fit(
   x=X_train_img, y=y_train_y,
   validation_data=(X_val_img, y_val_y),
   epochs=200,
   batch_size=32,
   verbose=2,
   callbacks=[early_stopping]
   #verbose=2
   #steps_per_epoch = X_train_num.shape[0]//batch_size,
   #validation_steps = X_train_num.shape[0]//batch_size,
)
print("TRAIN TIME: ", time.time()-t0)


cnn_modelY.save(images_folder+results_folder+'/cnn_modelY.h5')
tf.keras.backend.clear_session()
gc.collect()

def true_dist(y_pred, y_true):
   return np.mean(np.sqrt(
       np.square(np.abs(y_pred[:,0] - y_true[:,0]))
       + np.square(np.abs(y_pred[:,1] - y_true[:,1]))
       ))

# VALIDATION RESULTS
folder = images_folder+results_folder+"/predictions/cnn/validation/"
if not os.path.exists(folder):
   os.makedirs(folder)

start_time = time.time()
predX_val = cnn_modelX.predict(X_val_img)
print("PREDICTION TIME OF X (VALIDATION): ", time.time()-start_time)


Start_time = time.time()
predY_val = cnn_modelY.predict(X_val_img)
print("PREDICTION TIME OF Y (VALIDATION): ", time.time()-start_time)

predictions_valid = pd.DataFrame()
predictions_valid["realX"] = validation_x 
predictions_valid["realY"] = validation_y 


#predX_val = predX_val.reshape(-1)
predictions_valid["predX"] = predX_val
predictions_valid["predY"] = predY_val

predictions_valid.to_csv(folder+'preds_val.csv', index=False)


error_valid = true_dist(predictions_valid[["predX", "predY"]].to_numpy(), predictions_valid[["realX", "realY"]].to_numpy())
print(error_valid)


# RESULTS FOR TEST


folder = images_folder+results_folder+"/predictions/cnn/test/"
if not os.path.exists(folder):
   os.makedirs(folder)


start_time = time.time()
predictionX = cnn_modelX.predict(X_test_img)
print("PREDICTION TIME OF X (TEST): ", time.time()-start_time)


start_time = time.time()
predictionY = cnn_modelY.predict(X_test_img)
print("PREDICTION TIME OF Y (TEST): ", time.time()-start_time)

predictions_test = pd.DataFrame()
predictions_test["realX"] = test_x
predictions_test["realY"] = test_y


predictions_test["predX"] = predictionX
predictions_test["predY"] = predictionY

predictions_test.to_csv(folder+'preds_test.csv', index=False)


error_test = true_dist(predictions_test[["predX", "predY"]].to_numpy(), predictions_test[["realX", "realY"]].to_numpy())
print(error_test)


mae_x = mean_absolute_error(y_test_x, predictionX)
mse_x = mean_squared_error(y_test_x, predictionX)
rmse_x = mean_squared_error(y_test_x, predictionX, squared=False)
r2_x = r2_score(y_test_x, predictionX)


mae_y = mean_absolute_error(y_test_y, predictionY)
mse_y = mean_squared_error(y_test_y, predictionY)
rmse_y = mean_squared_error(y_test_y, predictionY, squared=False)
r2_y = r2_score(y_test_y, predictionY)

print("TEST X:")
print("Mean Absolute Error:", mae_x)
print("Mean Squared Error:", mse_x)
print("Root Mean Squared Error:", rmse_x)
print("R2 Score:", r2_x)
print()
print("TEST Y:")    
print("Mean Absolute Error:", mae_y)
print("Mean Squared Error:", mse_y)
print("Root Mean Squared Error:", rmse_y)
print("R2 Score:", r2_y)

results_filename = images_folder+results_folder+'evaluation_results_cnn.txt'
with open(results_filename, 'w') as results_file:
   results_file.write("Evaluation Metrics FOR X:\n")
   #results_file.write(f"Mean Absolute Percentage Error: {mape}\n")
   results_file.write(f"Mean Absolute Error: {mae_x}\n")
   results_file.write(f"Mean Squared Error: {mse_x}\n")
   results_file.write(f"Root Mean Squared Error: {rmse_x}\n")
   results_file.write(f"R2 Score: {r2_x}\n")
   results_file.write("\n")
   results_file.write("Evaluation Metrics FOR Y:\n")
   #results_file.write(f"Mean Absolute Percentage Error: {mape}\n")
   results_file.write(f"Mean Absolute Error: {mae_y}\n")
   results_file.write(f"Mean Squared Error: {mse_y}\n")
   results_file.write(f"Root Mean Squared Error: {rmse_y}\n")
   results_file.write(f"R2 Score: {r2_y}\n")
   results_file.write("\n")
   results_file.write(f"Error medio de validacion: {error_valid}\n")
   results_file.write(f"Error medio de test: {error_test}\n")