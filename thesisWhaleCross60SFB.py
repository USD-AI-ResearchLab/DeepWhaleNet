# !/usr/bin/env python
# coding: utf-8

#from modSpec import create_mod_spectrogram
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pickle
import tensorflow
from tensorflow import keras
from tensorflow.keras import regularizers
from keras.constraints import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import numpy as np
import pandas as pd
import random
import sklearn
#from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score, roc_curve, auc
from scipy import interp
#from audiomentations import *
#import matplotlib.pyplot as plt
from itertools import cycle
import ast
import time
# get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
#import cv2
import copy
import gc
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedKFold
from sklearn.datasets import make_multilabel_classification
from multiprocessing import Process, Manager
#from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit





#code found online for reinitializing the weights of a model in a method
def reset_weights(model):
  for layer in model.layers:
    if isinstance(layer, tensorflow.keras.Model):
      reset_weights(layer)
      continue
    for k, initializer in layer.__dict__.items():
      if "initializer" not in k:
        continue
      # find the corresponding variable
      var = getattr(layer, k.replace("_initializer", ""))
      var.assign(initializer(var.shape, var.dtype))

init =  HeNormal(seed=0)

def convolutional_block18(X, f, filters, stage, block, s=2):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(s, s), padding='same', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

    X_shortcut = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='same', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def identity_block18(X, f, filters, stage, block):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1 = filters

    X_shortcut = X
   
    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Add()([X, X_shortcut])# SKIP Connection
    X = Activation('relu')(X)

    return X


def ResNet18(input_shape=(input)):

    X_input = Input(input_shape)
    
    X = GaussianNoise(0.05)(X_input)

    X = RandomFlip(mode="horizontal", seed=None)(X)

    #X = RandomTranslation(height_factor = 0, width_factor = .1, fill_mode="nearest", interpolation="bilinear", seed=None, fill_value=0.0)(X)

    X = Conv2D(64, (7, 7), strides=(2, 2),padding="same", name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2),padding="same")(X)

    X = convolutional_block18(X, f=3, filters=64, stage=2, block='a', s=1)
    X = identity_block18(X, f=3, filters=64, stage=2, block='b')


    X = convolutional_block18(X, f=3, filters=128, stage=3, block='a', s=2)
    X = identity_block18(X, f=3, filters=128, stage=3, block='b')


    X = convolutional_block18(X, f=3, filters=256, stage=4, block='a', s=2)
    X = identity_block18(X, f=3, filters=256, stage=4, block='b')


    X = convolutional_block18(X, f=3, filters=512, stage=5, block='a', s=2)
    X = identity_block18(X, f=3, filters=512, stage=5, block='b')

    
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
    X = Flatten()(X)
    
    model = Model(inputs=X_input, outputs=X, name='ResNet18')

    return model

reg = None

def cnn(input_shape=(input)):

    X_input = Input(input_shape)
    
    X = GaussianNoise(0.025)(X_input)

    X = RandomFlip(mode="horizontal", seed=None)(X)

    X = RandomTranslation(height_factor = 0, width_factor = .1, fill_mode="nearest", interpolation="bilinear", seed=None, fill_value=0.0)(X)
    
    X = Conv2D(64, (7, 7), strides=(1, 1), padding="same", kernel_regularizer= reg)(X)
    X = BatchNormalization()(X)
    X = SpatialDropout2D(0.20, data_format='channels_last', )(X)
    X = Activation('relu')(X)

    X = Conv2D(64, (5, 5), strides=(1, 1), padding="same", kernel_regularizer= reg)(X)
    X = BatchNormalization()(X)
    X = SpatialDropout2D(0.20, data_format='channels_last', )(X)
    X = MaxPooling2D((2, 4), strides=(2, 4),padding="same",)(X)
    X = Activation('relu')(X)

    X = Conv2D(128, (3, 3), padding="same", kernel_regularizer= reg)(X)
    X = BatchNormalization()(X)
    X = Dropout(rate=0.20)(X)
    X = MaxPooling2D((2, 2), strides=(2, 2),padding="same")(X)
    X = Activation('relu')(X)
    
    X = Conv2D(256, (3, 3), padding="same", kernel_regularizer= reg)(X)
    X = BatchNormalization()(X)
    X = Dropout(rate=0.20)(X)
    X = Activation('sigmoid')(X)
    X = Flatten()(X)
    
    model = Model(inputs=X_input, outputs=X, name='crnn')

    return model


# Defining a function to create a dense block
def dense_block(x, num_layers, growth_rate):
  # Looping over the number of layers
  for i in range(num_layers):
    # Creating a bottleneck layer
    x1 = BatchNormalization()(x)
    x1 = Activation("relu")(x1)
    x1 = Conv2D(4 * growth_rate, (1, 1), padding="same")(x1)
    # Creating a convolution layer
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = Conv2D(growth_rate, (3, 3), padding="same")(x1)
    # Concatenating the input and the output
    x = Concatenate()([x, x1])
  # Returning the final output
  return x

# Defining a function to create a transition layer
def transition_layer(x, compression_factor):
  # Reducing the number of channels
  num_channels = int(x.shape[-1] * compression_factor)
  # Creating a batch normalization layer
  x = BatchNormalization()(x)
  # Creating a convolution layer
  x = Conv2D(num_channels, (1, 1), padding="same")(x)
  # Creating an average pooling layer
  x = AveragePooling2D((2, 2), strides=(2, 2))(x)
  # Returning the final output
  return x

def DenseNet(input_shape=(input)):

    num_blocks = 4 # The number of dense blocks
    num_layers = [4, 4, 4, 2]
    growth_rate = 8 # The growth rate of the network
    compression_factor = 0.5

    X_input = Input(input_shape)
    
    X = GaussianNoise(0.025)(X_input)

    X = RandomFlip(mode="horizontal", seed=None)(X)

    X = RandomTranslation(height_factor = 0, width_factor = .1, fill_mode="nearest", interpolation="bilinear", seed=None, fill_value=0.0)(X)

    X = Conv2D(64, (7, 7), strides=(1, 1), padding="same", kernel_regularizer= reg)(X)
    X = BatchNormalization()(X)
    #X = SpatialDropout2D(0.20, data_format='channels_last', )(X)
    X = Activation('relu')(X)

    for i in range(num_blocks):
        # Creating a dense block
        X = dense_block(X, num_layers[i], growth_rate)
        # Creating a transition layer if it is not the last block
        if i != num_blocks - 1:
            X = transition_layer(X, compression_factor)

    X = BatchNormalization()(X)
    X = Activation("sigmoid")(X)
    X = Flatten()(X)

    model = Model(inputs=X_input, outputs=X, name='DenseNet')

    return model


def runTrain(L):

    physical_devices = tensorflow.config.list_physical_devices('gpu') 
    for gpu_instance in physical_devices: 
        tensorflow.config.experimental.set_memory_growth(gpu_instance, True)
        
    # Get the list of GPUs
    gpus = tensorflow.config.list_physical_devices('GPU')
    
    # Set the memory limit for each GPU
    for gpu in gpus:
      tensorflow.config.set_logical_device_configuration(
        gpu,
        [tensorflow.config.LogicalDeviceConfiguration(memory_limit=75 * 1024)]
      )

    resultsloss = L[0]
    resultsacc = L[1]
    X_train = L[2]
    y_train = L[3]
    X_test = L[4]
    y_test = L[5]
    count = L[6]

    model = Sequential()
    input_shape=(128, 256, 1)
    model.add(ResNet18(input_shape=input_shape))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.20))
    model.add(Activation('sigmoid'))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    initial_learning_rate = 0.01
    lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.5,
        staircase=False)

    model.compile(optimizer = Adam(learning_rate = lr_schedule), loss = 'binary_crossentropy' , metrics = [tensorflow.keras.metrics.BinaryAccuracy()])
    
    time.sleep(10.00)
    #clear out old memory
    tensorflow.keras.backend.clear_session()
    #re-initialize the models weights
    reset_weights(model)

    #resetting the new checkpoint for this run through
    model_checkpoint = [
        
        ModelCheckpoint(filepath = './checks/thesisWhaleRes60SFB25664' + str(count) + ".hdf5", monitor='val_binary_accuracy',verbose=1, save_best_only=True, mode = 'max'),
        #tensorflow.keras.callbacks.EarlyStopping(
        #    monitor="val_auc",
        #    min_delta=.0001,
        #    patience=13,
        #    verbose=0,
        #    mode="max",
        #    baseline=None,
        #    restore_best_weights=False,
        #),
        #tensorflow.keras.callbacks.ReduceLROnPlateau(
        #    monitor="val_auc",
        #    factor=0.5,
        #    patience=5,
        #    verbose=0,
        #    mode="max",
        #    min_delta=0.0001,
        #    cooldown=0,
        #    min_lr=0
        #)
        
    ]

    #executing the train function
    history = model.fit(
          x=X_train,
          y=y_train,
          epochs=50,
          batch_size = 64,
          validation_data = (X_test, y_test),
          callbacks=[model_checkpoint],
          class_weight=None,
          max_queue_size=10,
          shuffle = True)
    
    
    time.sleep(2.00)
    tensorflow.keras.backend.clear_session()
    
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('Model binary accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('./results/Accuracy60S12832' + str(count) +'.png')
    
    plt.clf()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('./results/Loss60S12832' + str(count) +'.png')

    # loading the best checkpoint for the train/validation
    model.load_weights('./checks/thesisWhaleRes60SFB25664' + str(count) + '.hdf5')

    #clear out old memory
    time.sleep(2.00)
    tensorflow.keras.backend.clear_session()

    #getting the best validation scores
    score = model.evaluate(
        x=X_test,
        y=y_test,
        batch_size = 64)
        #getting the total real negatives in the validation data
    #y_testN = [i for i in y_val if i < .5]

    #adding this interations results
    resultsloss.append(score[0])
    resultsacc.append(score[1])

    count+=1

    L[0] = resultsloss
    L[1] = resultsacc
    L[6] = count





if __name__ == "__main__":

    

    infile = open('./dataframes/allDatasets250_60FBWhale.pkl','rb')
    data = pickle.load(infile)
    infile.close()

    data = data.sample(frac = 1, random_state=10)


    dataset = [] # Train Dataset
    vdataset =[]
    #iterating over valid train/validation data
    count = 0
    countn = 0
    for row in data.itertuples():
        
        if row.dataset == 'casey2017':
            continue

        print(count)
        arrList=[]
        #if row.source !='coswara':
        #    continue
    
        y, label = row.array, np.array(row.label)

        if len(y) != 16384:
            continue

        if 1 not in label:
            countn+=1
            if countn>77500:
                continue
    
        lArr = []
        countm = 0
        for x in label:
                  
            lArr.append(x)

        stft = np.abs(librosa.stft(y = y, n_fft = 256, hop_length = 64))**2
    
        ms2 = stft/stft.max()
        ms_DB = librosa.power_to_db(S = ms2, ref = 0)
        ms_DB = ms_DB - 20
        ms_DB = ms_DB/ms_DB.max()
        ms_DB = ms_DB[1:,1:]
        ms_DB = np.reshape(ms_DB,(128,256,1))
    
        count+=1
    
        dataset.append( (ms_DB, np.array(lArr)) )

    data_X, data_y = zip(*dataset)
    # Train/Validation Dataset

    data_X, data_y = np.array(data_X), np.array(data_y)


    print(data_y)
    #initializing variables and sets for the train/validation loop


    

    label = []

    for x in data_y:
        if x[0] == 1:
            label.append(0)
        elif x[1] == 1:
            label.append(1)

    label = np.array(label)

    count = 0
    resultsloss = []
    resultsacc = []
    train = []
    val = []
    trainY = []
    valY = []

    #manager = Manager()
    #lst = manager.list()
    
    lst = []
    lst.append(resultsloss)
    lst.append(resultsacc)
    lst.append(train)
    lst.append(val)
    lst.append(trainY)
    lst.append(valY)
    lst.append(count)


    #begining train/validation loop
    skf = RepeatedKFold(n_splits = 5, random_state = 10, n_repeats=2)
    #skf.get_n_splits(data_X, data_y)

    for trainIdx, testIdx in skf.split(data_X, data_y):
        if lst[6] >0 :
          break
    
        gc.collect()
        #if lst[6] < 2:
        #    lst[6] += 1
        #    continue

        X_train = data_X[trainIdx]
        X_test = data_X[testIdx]
        y_train = []
        y_test = []

        
        X_train = np.array([x.reshape( (128,256,1 ) ) for x in X_train])
        X_test = np.array([x.reshape( (128,256,1 ) ) for x in X_test])
        y_train = data_y[trainIdx]
        y_test = data_y[testIdx]

        print(X_train.shape)
        print(y_train.shape)

        lst[2] = X_train
        lst[3] = y_train
        lst[4] = X_test
        lst[5] = y_test

        #p = Process(target=runTrain, args=[lst])
        #p.start()
        #p.join()
        runTrain(lst)

        #just cuz
        continue

    resultsloss = lst[0]
    resultsacc = lst[1]

    #convert results of train/validation to a dataframe
    df_results = pd.DataFrame(data={"Loss":resultsloss, "Acc":resultsacc})
    #df_results.to_csv('./Virufy_Train_Data1.csv')
    df_results = df_results[['Loss', 'Acc']]

    #adding the average of the results dataframe
    resultsloss.append('Avg')
    resultsacc.append('Avg')
    #resultsauc.append('Avg')
    #resultssens.append('Avg')
    #resultsspec.append('Avg')

    resultsloss.append(df_results['Loss'].mean())
    resultsacc.append(df_results['Acc'].mean())
    #resultsauc.append(df_results['AUC'].mean())
    #resultssens.append(df_results['Sens'].mean())
    #resultsspec.append(df_results['Spec'].mean())


    #recreating the results dataframe with the average and test data
    df_results = pd.DataFrame(data={"Loss":resultsloss, "Acc":resultsacc})

    #sending to a CSV
    df_results.to_csv('./results/whaleCross.csv')
    
    
    