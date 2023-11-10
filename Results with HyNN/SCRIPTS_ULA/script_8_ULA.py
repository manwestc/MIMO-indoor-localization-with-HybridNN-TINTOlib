# python version: 3.8.3

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
#import cv2
import gc
import matplotlib.pyplot as plt
#import openslide
#from openslide.deepzoom import DeepZoomGenerator
import tifffile as tifi
import sklearn
import tensorflow as tf
import seaborn as sns
from PIL import Image
import time


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import vgg16, vgg19, resnet50, mobilenet, inception_resnet_v2, densenet, inception_v3, xception, nasnet, ResNet152V2
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, InputLayer, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, Adamax
from tensorflow.keras import layers, models, Model
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.layers import Input, Activation,MaxPooling2D, concatenate

from torchmetrics import MeanAbsolutePercentageError
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers  import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from imblearn.over_sampling import RandomOverSampler

#Models of TINTOlib
from TINTOlib.tinto import TINTO
from TINTOlib.supertml import SuperTML
from TINTOlib.igtd import IGTD
from TINTOlib.refined import REFINED
from TINTOlib.barGraph import BarGraph
from TINTOlib.distanceMatrix import DistanceMatrix
from TINTOlib.combination import Combination

df = pd.read_csv('ULA_lab_LoS_8.csv')

#Select the model and the parameters
problem_type = "regression"
pixel = 35
image_model = TINTO(problem= problem_type,pixels=pixel,blur=True)


images_folder = "images_8antennas_ULA"
results_folder = "/results/"


# * NORMALIZE DATASET

# Select all the attributes to normalize
columns_to_normalize = df.columns[:-2]

# Normalize between 0 and 1
df_normalized = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())

# Combine the attributes and the label
#df_normalized = pd.concat([df_normalized, df[df.columns[-1]]], axis=1)
df_normalized = pd.concat([df_normalized, df[df.columns[-2]], df[df.columns[-1]]], axis=1)

#df_normalized.head(2)

#df.iloc[:,:-1].head(2)


#Generate the images
#image_model.generateImages(df, images_folder)

if not os.path.exists(images_folder):
    print("generating images...")
    image_model.generateImages(df.iloc[:,:-1], images_folder)
    print("Images generated")
    

if not os.path.exists(images_folder+results_folder):
    os.makedirs(images_folder+results_folder)
    
img_paths = os.path.join(images_folder,problem_type+".csv")


imgs = pd.read_csv(img_paths) 

imgs["images"]= images_folder + "/" + imgs["images"] 

imgs["images"] = imgs["images"].str.replace("\\","/")

#combined_dataset = pd.concat([imgs,df_normalized],axis=1)

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

# For 3 canal (RGB)
X_train_img = np.array([cv2.resize(cv2.imread(img),(pixel,pixel)) for img in X_train["images"]])
X_val_img = np.array([cv2.resize(cv2.imread(img),(pixel,pixel)) for img in X_val["images"]])
X_test_img = np.array([cv2.resize(cv2.imread(img),(pixel,pixel)) for img in X_test["images"]])
    

validacion_x = y_val_x
test_x = y_test_x
validacion_y = y_val_y
test_y = y_test_y

shape = len(X_train_num.columns)


from keras.layers import AveragePooling2D, Concatenate

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

ff_model = Model(inputs = ff_inputs, outputs = merged_tabular)

# * CNN

#input
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


#out = Dense(n_class, activation='softmax')(out)

cnn_model = Model(input_shape, out)


combinedInput = concatenate([ff_model.output, cnn_model.output])
x = Dense(64, activation="relu")(combinedInput)
x = Dense(32, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dense(1, activation="linear")(x)

modelX = Model(inputs=[ff_model.input, cnn_model.input], outputs=x)

#from keras.utils import plot_model
#model.summary()
#plot_model(model)#, to_file='convolutional_neural_network.png')
from tensorflow_addons.metrics import RSquare

METRICS = [
    tf.keras.metrics.MeanSquaredError(name = 'mse'),
    tf.keras.metrics.MeanAbsoluteError(name = 'mae'),
    #tf.keras.metrics.R2Score(name = 'r2'),
    RSquare(name='r2_score'),
    tf.keras.metrics.RootMeanSquaredError(name = 'rmse')
]

#opt = Adam(learning_rate=1e-5)
opt = Adam()
modelX.compile(
    loss="mse",
    optimizer=opt,
    metrics = METRICS
)

# Define EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


t0 = time.time()

model_history=modelX.fit(
    x=[X_train_num, X_train_img], y=y_train_x,
    validation_data=([X_val_num, X_val_img], y_val_x),
    epochs=200,
    batch_size=32,
    verbose=2,
    callbacks=[early_stopping]
    #verbose=2
    #steps_per_epoch = X_train_num.shape[0]//batch_size,
    #validation_steps = X_train_num.shape[0]//batch_size,
)
print("TRAIN TIME: ", time.time()-t0)

#print(model_history.history.keys())

modelX.save(images_folder+'/modelX_8.h5')

# RESULTS

plt.plot(model_history.history['loss'], color = 'red', label = 'loss')
plt.plot(model_history.history['val_loss'], color = 'green', label = 'val loss')
plt.legend(loc = 'upper right')
plt.savefig(images_folder+results_folder+'loss_graphX.png')
plt.clf()


plt.plot(model_history.history['mse'], color = 'red', label = 'mse')
plt.plot(model_history.history['val_mse'], color = 'green', label = 'val mse')
plt.legend(loc = 'upper right')
plt.savefig(images_folder+results_folder+'mse_graphX.png')
plt.clf()


plt.plot(model_history.history['mae'], color = 'red', label = 'mae')
plt.plot(model_history.history['val_mae'], color = 'green', label = 'val mae')
plt.legend(loc = 'upper right')
plt.savefig(images_folder+results_folder+'mae_graphX.png')
plt.clf()

# //////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////
#   *                           MODEL FOR Y
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

ff_model = Model(inputs = ff_inputs, outputs = merged_tabular)


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


#out = Dense(n_class, activation='softmax')(out)

cnn_model = Model(input_shape, out)


combinedInput = concatenate([ff_model.output, cnn_model.output])
x = Dense(64, activation="relu")(combinedInput)
x = Dense(32, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x = Dense(8, activation="relu")(x)
x = Dense(1, activation="linear")(x)

modelY = Model(inputs=[ff_model.input, cnn_model.input], outputs=x)

from tensorflow_addons.metrics import RSquare

METRICS = [
    tf.keras.metrics.MeanSquaredError(name = 'mse'),
    tf.keras.metrics.MeanAbsoluteError(name = 'mae'),
    #tf.keras.metrics.R2Score(name='r2_score'),
    RSquare(name='r2_score'),
    tf.keras.metrics.RootMeanSquaredError(name='rmse')
]

opt = Adam(learning_rate=1e-4)
#opt = Adam()
modelY.compile(
    loss="mse",
    optimizer=opt,
    metrics = METRICS
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

t0 = time.time()

model_history=modelY.fit(
    x=[X_train_num, X_train_img], y=y_train_y,
    validation_data=([X_val_num, X_val_img], y_val_y),
    epochs=200,
    batch_size=32,
    verbose=2,
    callbacks=[early_stopping]
    #verbose=2
    #steps_per_epoch = X_train_num.shape[0]//batch_size,
    #validation_steps = X_train_num.shape[0]//batch_size,
)
print("TRAIN TIME: ", time.time()-t0)

modelY.save(images_folder+'/modelY_8.h5')

# RESULTS

plt.plot(model_history.history['loss'], color = 'red', label = 'loss')
plt.plot(model_history.history['val_loss'], color = 'green', label = 'val loss')
plt.legend(loc = 'upper right')
plt.savefig(images_folder+results_folder+'loss_graphY.png')
plt.clf()


plt.plot(model_history.history['mse'], color = 'red', label = 'mse')
plt.plot(model_history.history['val_mse'], color = 'green', label = 'val mse')
plt.legend(loc = 'upper right')
plt.savefig(images_folder+results_folder+'mse_graphY.png')
plt.clf()


plt.plot(model_history.history['mae'], color = 'red', label = 'mae')
plt.plot(model_history.history['val_mae'], color = 'green', label = 'val mae')
plt.legend(loc = 'upper right')
plt.savefig(images_folder+results_folder+'mae_graphY.png')
plt.clf()


def true_dist(y_pred, y_true):
    return np.mean(np.sqrt(
        np.square(np.abs(y_pred[:,0] - y_true[:,0]))
        + np.square(np.abs(y_pred[:,1] - y_true[:,1]))
        ))

# VALIDATION RESULTS
folder = images_folder+"/predictions/ULA/8antennas/validation/"
if not os.path.exists(folder):
    os.makedirs(folder)

start_time = time.time()
predX_val = modelX.predict([X_val_num,X_val_img])
print("PREDICTION TIME OF X (VALIDATION): ", time.time()-start_time)

Start_time = time.time()
predY_val = modelY.predict([X_val_num,X_val_img])
print("PREDICTION TIME OF Y (VALIDATION): ", time.time()-start_time)

predicciones_valid = pd.DataFrame()
predicciones_valid["realX"] = validacion_x 
predicciones_valid["realY"] = validacion_y 

#predX_val = predX_val.reshape(-1)
predicciones_valid["predX"] = predX_val
predicciones_valid["predY"] = predY_val

predicciones_valid.to_csv(folder+'preds_val.csv', index=False)

error_valid = true_dist(predicciones_valid[["predX", "predY"]].to_numpy(), predicciones_valid[["realX", "realY"]].to_numpy())
print(error_valid)



# RESULTS FOR TEST

start_time = time.time()
predictionX = modelX.predict([X_test_num,X_test_img])
print("PREDICTION TIME OF X (TEST): ", time.time()-start_time)

start_time = time.time()
predictionY = modelY.predict([X_test_num,X_test_img])
print("PREDICTION TIME OF Y (TEST): ", time.time()-start_time)

folder = images_folder+"/predictions/ULA/8antennas/test/"
if not os.path.exists(folder):
    os.makedirs(folder)

predicciones_test = pd.DataFrame()
predicciones_test["realX"] = test_x
predicciones_test["realY"] = test_y

#predX_test = predictionX.reshape(-1)
predicciones_test["predX"] = predictionX
predicciones_test["predY"] = predictionY

predicciones_test.to_csv(folder+'preds_test.csv', index=False)

error_test = true_dist(predicciones_test[["predX", "predY"]].to_numpy(), predicciones_test[["realX", "realY"]].to_numpy())

print(error_test)


mae_x = mean_absolute_error(y_test_x, predictionX)
mse_x = mean_squared_error(y_test_x, predictionX)
rmse_x = mean_squared_error(y_test_x, predictionX, squared=False)
r2_x = r2_score(y_test_x, predictionX)

mae_y = mean_absolute_error(y_test_y, predictionY)
mse_y = mean_squared_error(y_test_y, predictionY)
rmse_y = mean_squared_error(y_test_y, predictionY, squared=False)
r2_y = r2_score(y_test_y, predictionY)

# Save evaluation metrics to a text file
results_filename = images_folder+results_folder+'evaluation_results.txt'
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

# Print the evaluation metrics
#print("Mean Absolute Percentage Error:", mape)
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