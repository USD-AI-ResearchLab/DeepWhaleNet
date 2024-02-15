# !/usr/bin/env python
# coding: utf-8

#from modSpec import create_mod_spectrogram
import matplotlib.ticker as ticker
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
from sklearn.metrics import *
import librosa
import librosa.display
import numpy as np
import pandas as pd
import random
import sklearn
#from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score, roc_curve, auc
from scipy import interp
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

physical_devices = tensorflow.config.list_physical_devices('gpu') 
for gpu_instance in physical_devices: 
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)





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

    X_shortcut = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
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
    
    X = GaussianNoise(0.025)(X_input)

    X = RandomFlip(mode="horizontal", seed=None)(X)

    X = RandomTranslation(height_factor = 0, width_factor = .1, fill_mode="nearest", interpolation="bilinear", seed=None, fill_value=0.0)(X)

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

model = Sequential()
input_shape=(128, 256, 1)

model.add(ResNet18(input_shape))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dropout(rate=0.20))
model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.summary()

initial_learning_rate = 0.01
lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=6590,
    decay_rate=0.9,
    staircase=False)

infile = open('./dataframes/allDatasets250_60FBWhale.pkl','rb')
data = pickle.load(infile)
infile.close()

data = data.query("dataset == 'casey2017'")

# loading the best checkpoint for the train/validation
model.load_weights('./checks/thesisWhaleRes60SFB256640.hdf5')


boot = []
probs = []
pred = []
dataset = [] # Train Dataset
vdataset =[]
#iterating over valid train/validation data
count = 0
for row in data.itertuples():
    print(row[0])
    arrList=[]
    
    
    
    y, label = row.array, np.array(row.label)

    if len(y) != 16384:
        data.drop(row[0],axis=0,inplace=True)
        continue
    
    lArr=[]
    
    for x in label:
      if x == 1:
        lArr.append(1)
      else:
        lArr.append(0)
    
    stft = np.abs(librosa.stft(y = y, n_fft = 256, hop_length = 64))**2
    
    ms2 = stft/stft.max()
    ms_DB = librosa.power_to_db(S = ms2, ref = 0)
    ms_DB = ms_DB - 20
    ms_DB = ms_DB/ms_DB.max()
    ms_DB = ms_DB[1:,1:]
    ms_DB = np.reshape(ms_DB,(128,256,1))

    dataset.append( (ms_DB, np.array(lArr)) )

    count+=1 

data_X, data_y = zip(*dataset)

#initializing variables and sets for the train/validation loop

count = 0
resultsloss = []
resultsacc = []
resultsacc = []
resultsauc = []
resultssens = []
resultsspec = []


#preparing the train/validation data
#count = 0   
#X_train, y_train = zip(*dataset)
X_test = np.array([x.reshape( (128, 256,1 ) ) for x in data_X])
y_test = np.array(data_y)
probss = model.predict(X_test)
probs = []

for i in range(len(probss)):
  lArr = []
  for j in range(2):
    lArr.append(probss[i,j])
  probs.append(np.array(lArr))


pred = []

for x in probs:
    lArr = []
    for y in x:
      if y > .5:
        lArr.append(1)
      else:
        lArr.append(0)
        
    pred.append(np.array(lArr))

data['probs'] = probs
data['pred'] = pred

initial_learning_rate = 0.01
lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=6590,
    decay_rate=0.9,
    staircase=False)

model.compile(optimizer = Adamax(learning_rate = lr_schedule), loss = 'binary_crossentropy', metrics = ['Accuracy'])

score = model.evaluate(
        x=X_test,
        y=y_test,
        batch_size = 64)

y_pred = np.array(pred)

print(classification_report(y_test,y_pred ))

datacopy = copy.deepcopy(data)
datacopy2 = copy.deepcopy(data)
datacopy3 = copy.deepcopy(data)

datacopy2['rel'] = False
datacopy2['probs'] = list([x.reshape((2))[0] for x in probss])
datacopy3['rel'] = False
datacopy3['probs'] = list([x.reshape((2))[1] for x in probss])


data2 = pd.read_csv( './casey2017/Casey2017.Bm.D.selections.txt', sep = '\t' )
data3 = pd.read_csv( './casey2017/Casey2017.Bm.Ant-A.selections.txt', sep = '\t' )
data4 = pd.read_csv( './casey2017/Casey2017.Bm.Ant-B.selections.txt', sep = '\t' )
data5 = pd.read_csv( './casey2017/Casey2017.Bm.Ant-Z.selections.txt', sep = '\t' )

data2 = pd.concat([data2,data3,data4,data5])


data2['hit'] = False

remove = []
hop = 8192
ss = 16384
for row in data2.itertuples():
   for chunk in data.itertuples():
        if (('wav\\' + str(row[4]))!=('wav\\' + str(row[5])) ):
            if (('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[8]) or (('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[9] and ((chunk.chunks*hop)+ss)*4 >= row[9]):
                remove.append(chunk[0])
                if chunk.pred[0] == 1:
                    data2.loc[row[0],'hit'] = True
        else:
            if ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[9]:
                remove.append(chunk[0])
                if chunk.pred[0] == 1:
                    data2.loc[row[0],'hit'] = True
            elif ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[8] and ((chunk.chunks*hop)+ss)*4 <= row[9] or ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 >= row[8] and (chunk.chunks*hop)*4 <= row[9] and ((chunk.chunks*hop)+ss)*4 >= row[9]:
                remove.append(chunk[0])
                if chunk.pred[0] == 1:
                    data2.loc[row[0],'hit'] = True

data2.to_csv('./results/thesisWhale60SBFB25664Hits.csv')

remove = np.unique(remove, axis=0)

data.drop(np.array(remove), axis = 0 , inplace = True)
data.index = range(len(data))

for x in data.itertuples():
      if x[0]+1 >= len(data):
        break
      t = True
      count = 0
      while(t):
        if count > 1:
              break
        if x[0]+1 < len(data) and isinstance(data.loc[(int(x[0]+1)),'pred'],(list,np.ndarray)) and isinstance(x.pred,(list,np.ndarray)):
          if x.pred[0] == 1 and (data.loc[(int(x[0]+1)),'pred'][0] == 1):
            data.drop(int(x[0]+1), axis = 0 , inplace = True)
            data.index = range(len(data))
            count+=1
          else:
            t = False
        else:
            t = False



for row in data2.itertuples():
   for chunk in datacopy2.itertuples():
        if (('wav\\' + str(row[4]))!=('wav\\' + str(row[5])) ):
            if (('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[8]) or (('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[9] and ((chunk.chunks*hop)+ss)*4 >= row[9]):
                datacopy2.loc[chunk[0],'rel'] = True
        else:
            if ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[9]:
                datacopy2.loc[chunk[0],'rel'] = True
            elif ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[8] and ((chunk.chunks*hop)+ss)*4 <= row[9] or ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 >= row[8] and (chunk.chunks*hop)*4 <= row[9] and ((chunk.chunks*hop)+ss)*4 >= row[9]:
                datacopy2.loc[chunk[0],'rel'] = True
                
datacopy2.sort_values('probs', ascending = False, inplace = True)
datacopy2.index = range(len(datacopy2))

rrs = []
count = 0
recCount = 0
for x in datacopy2.itertuples():
        count+=1
        if x.rel == True:
            recCount+=1
            rrs.append(recCount/count)

average = sum(rrs) / len(rrs)      

fP = 0
for x in data.itertuples():
    if isinstance(x.pred,(list,np.ndarray)):
        if x.pred[0] == 1:
            fP +=1
          
tP = len(data2.query('hit == True'))
totalP = len(data2)
recall = (tP/totalP)
prec = (tP/(tP+fP))

print('Recall:', recall)
print('Precision:', prec)
print('F1_Score', ((2*(recall*prec))/(recall+prec)))
print("MAP:", average)

data2 = pd.read_csv( './casey2017/Casey2017.Bp.20Hz.selections.txt', sep = '\t' )
data3 = pd.read_csv( './casey2017/Casey2017.Bp.20Plus.selections.txt', sep = '\t' )
data4 = pd.read_csv( './casey2017/Casey2017.Bp.Downsweep.selections.txt', sep = '\t' )

data2 = pd.concat([data2,data3,data4])


data2['hit'] = False

remove = []
for row in data2.itertuples():
   for chunk in datacopy.itertuples():
        if (('wav\\' + str(row[4]))!=('wav\\' + str(row[5])) ):
            if (('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[8]) or (('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[9] and ((chunk.chunks*hop)+ss)*4 >= row[9]):
                remove.append(chunk[0])
                if chunk.pred[1] == 1:
                    data2.loc[row[0],'hit'] = True
        else:
            if ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[9]:
                remove.append(chunk[0])
                if chunk.pred[1] == 1:
                    data2.loc[row[0],'hit'] = True
            elif ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[8] and ((chunk.chunks*hop)+ss)*4 <= row[9] or ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 >= row[8] and (chunk.chunks*hop)*4 <= row[9] and ((chunk.chunks*hop)+ss)*4 >= row[9]:
                remove.append(chunk[0])
                if chunk.pred[1] == 1:
                    data2.loc[row[0],'hit'] = True

data2.to_csv('./results/thesisWhale60SBFF25664Hits.csv')

remove = np.unique(remove, axis=0)

datacopy.drop(np.array(remove), axis = 0 , inplace = True)
datacopy.index = range(len(datacopy))

for x in datacopy.itertuples():
      if x[0]+1 >= len(datacopy):
        break
      t = True
      while(t):
        if count > 1:
              break
        if x[0]+1 < len(datacopy) and isinstance(datacopy.loc[(int(x[0]+1)),'pred'],(list,np.ndarray)) and isinstance(x.pred,(list,np.ndarray)):
          if x.pred[1] == 1 and (datacopy.loc[(int(x[0]+1)),'pred'][1] == 1):
            datacopy.drop(int(x[0]+1), axis = 0 , inplace = True)
            datacopy.index = range(len(datacopy))
            count+=1
          else:
            t = False
        else:
            t = False


for row in data2.itertuples():
   for chunk in datacopy3.itertuples():
        if (('wav\\' + str(row[4]))!=('wav\\' + str(row[5])) ):
            if (('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[8]) or (('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[9] and ((chunk.chunks*hop)+ss)*4 >= row[9]):
                datacopy3.loc[chunk[0],'rel'] = True
        else:
            if ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[9]:
                datacopy3.loc[chunk[0],'rel'] = True
            elif ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 <= row[8] and ((chunk.chunks*hop)+ss)*4 >= row[8] and ((chunk.chunks*hop)+ss)*4 <= row[9] or ('wav\\' + str(row[4])) == str(chunk.fileName) and chunk.chunks*hop*4 >= row[8] and (chunk.chunks*hop)*4 <= row[9] and ((chunk.chunks*hop)+ss)*4 >= row[9]:
                datacopy3.loc[chunk[0],'rel'] = True
                
datacopy3.sort_values('probs', ascending = False, inplace = True)
datacopy3.index = range(len(datacopy3))

rrs = []
count = 0
recCount = 0
for x in datacopy3.itertuples():
        count+=1
        if x.rel == True:
            recCount+=1
            rrs.append(recCount/count)

average = sum(rrs) / len(rrs)

fP = 0
for x in datacopy.itertuples():
    if isinstance(x.pred,(list,np.ndarray)):
        if x.pred[1] == 1:
            fP +=1          
            
tP = len(data2.query('hit == True'))
totalP = len(data2)
recall = (tP/totalP)
prec = (tP/(tP+fP))

print('Recall:', recall)
print('Precision:', prec)
print('F1_Score', ((2*(recall*prec))/(recall+prec)))
print("MAP:", average)

