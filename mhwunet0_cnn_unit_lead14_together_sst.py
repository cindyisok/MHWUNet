# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:58:40 2023

@author: Cindy
"""

import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, Dense, ConvLSTM2D, MaxPooling2D, Dropout, UpSampling2D, \
    concatenate, BatchNormalization, add, Activation, multiply, Reshape, ZeroPadding2D, Cropping2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import tensorflow as tf
from mpl_toolkits.basemap import Basemap
import math
import scipy.io as io

# our own data
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
tf.random.set_seed(seed)

data = io.loadmat('/home/jingzhao/mhw_forecast/data/UKMO/data_v1/ssta_OISST_1.5x1.5_smooth_82_22_interp2_UKMO_no_land_ENSO.mat')
sst_train = np.expand_dims(data['ssta'], 3)[:,20:100,:2543,:]
print(sst_train.shape)

data1 = h5py.File('/home/jingzhao/mhw_forecast/data/UKMO/data_v1/ssta_UKMO_2016_2022_no_land_ENSO.mat','r')
t2_train = np.expand_dims(np.transpose(data1['ssta_UKMO']),4)[:,20:100,:,:2543]
print(t2_train.shape)

data2 = h5py.File('/home/jingzhao/mhw_forecast/data/UKMO/data_v1/mask_OISST_2016_2022_no_prolonged_ENSO.mat')
Label_train = np.expand_dims(np.transpose(data2['mask_OISST']), 4)[:,20:100,:,:,:]
print(Label_train.shape)

Label_train_categor = tf.keras.utils.to_categorical(Label_train, 2)#.transpose([2,3,0,1])
print(Label_train_categor.shape)
print("============Loading data end===============")
del Label_train

print("==============Beginning================")
def FirstResUnit(nf,ker,inputs):
    conv1 = Conv2D(nf, ker, padding="same", kernel_initializer='he_normal', use_bias=False)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(nf, ker, padding="same", kernel_initializer='he_normal', use_bias=False)(conv1)
   # conv2 = BatchNormalization()(conv2)
    shortcut = Conv2D(filters=nf, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
   # shortcut = BatchNormalization()(shortcut)
    return add([conv2,shortcut])
def ResUnit(nf,ker,inputs, drop=0.5):
    conv1 = BatchNormalization()(inputs)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(drop)(conv1)
    conv2 = Conv2D(nf, ker, padding="same", kernel_initializer='he_normal', use_bias=False)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(drop)(conv2)
    conv2 = Conv2D(nf, ker, padding="same", kernel_initializer='he_normal', use_bias=False)(conv2)
    return add([conv2,inputs])
def ResUnitDecoder(nf,ker,inputs,drop=0.5):
    conv1 = BatchNormalization()(inputs)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(drop)(conv1)
    conv2 = Conv2D(nf, ker, padding="same", kernel_initializer='he_normal', use_bias=False)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(drop)(conv2)
    conv2 = Conv2D(nf, ker, padding="same", kernel_initializer='he_normal', use_bias=False)(conv2)
    ###
    shortcut = Conv2D(nf, 1, padding="same", kernel_initializer='he_normal', use_bias=False)(inputs)
    return add([conv2,shortcut])

def attention_block(x, gating_signal):
    theta_x = Conv2D(filters=x.shape[-1], kernel_size=1, strides=1, padding='same')(x)
    phi_g = Conv2D(filters=gating_signal.shape[-1], kernel_size=1, strides=1, padding='same')(gating_signal)
            
    f = Activation('relu')(theta_x + phi_g)
                    
    psi_f = Conv2D(filters=1, kernel_size=1, strides=1, padding='same')(f)
    sigmoid_psi_f = Activation('sigmoid')(psi_f)
                                
    attention_mul = tf.keras.layers.Multiply()([x, sigmoid_psi_f])
    return attention_mul
def CBAM(input_sensor, filters):
    x1 = GlobalAveragePooling2D(data_format = "channels_last")(input_sensor)
    x2 = GlobalMaxPooling2D(data_format = "channels_last")(input_sensor)

    x1 = Dense(4*filters, activation='relu')(x1)
    x1 = Dense(input_sensor.shape[-1])(x1)
    x2 = Dense(4*filters, activation='relu')(x2)
    x2 = Dense(input_sensor.shape[-1])(x2)

    x1 = K.expand_dims(x1, axis=1)
    x1 = K.expand_dims(x1, axis=1)
    x2 = K.expand_dims(x2, axis=1)
    x2 = K.expand_dims(x2, axis=1)

    x = add([x1,x2])
    x = Activation('sigmoid')(x)
    x0 = multiply([input_sensor,x])

    x1 = K.mean(x0, axis=-1)
    x2 = K.max(x0, axis=-1)

    x1 = K.expand_dims(x1, axis=-1)
    x2 = K.expand_dims(x2, axis=-1)

    x = concatenate([x1,x2], axis=-1)

    x = Conv2D(1, (7,7), padding='same', data_format = "channels_last")(x)
    x = Activation('sigmoid')(x)
    x = multiply([x0,x])
    return x
#############PARAMETER INPUT##################
time_sequence = 2
# time_predict = 7
height = 240
width = 80
nf = 32
ker = 3
num_responses = 2
############INPUT LAYER############

img_input = Input(shape=(height,width,16))
###############################ENCODER#############################
#preprocess = ZeroPadding2D(padding=4)(img_input)
conv1 = FirstResUnit(nf,ker,img_input)
pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

conv2 = ResUnit(nf,ker,pool1)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

conv3 = ResUnit(nf,ker,pool2)
pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

###################center#####################

convC = ResUnit(nf,ker,pool3)

cbam = CBAM(convC, 20)
#attention_gating = Conv2D(nf, 1, activation='relu')(convC)
#attention_mul = attention_block(convC, attention_gating)
convC = ResUnit(nf,ker,cbam)

###########################DECODER##########################

up3 = concatenate([UpSampling2D((2,2))(convC), conv3])
decod3 = ResUnitDecoder(nf,ker,up3)

up2 = concatenate([UpSampling2D((2,2))(decod3), conv2])
decod2 = ResUnitDecoder(nf,ker,up2)

up1 = concatenate([UpSampling2D((2,2))(decod2), conv1])
decod1 = ResUnitDecoder(nf,ker,up1)
#outputs = Cropping2D(cropping=4)(decod1)
####################################### Segmentation Layer
outputs = Conv2D(num_responses*15, (1, 1), padding="valid", use_bias=False)(decod1)
outputs = Reshape((height, width, 15, num_responses))(outputs)
outputs = Activation('softmax')(outputs)

mhwunet = Model(img_input, outputs)
mhwunet.summary()

###loss function

smooth = 1.  # to avoid zero division

def similarity(y_true, y_pred):
    y_true_1day = y_pred[:,:,:,0,1]
    y_true_14day = y_pred[:,:,:,13,1]
    intersection_mhw = K.sum(y_true_1day * y_true_14day)
    return intersection_mhw/K.sum(y_true_1day)

def difference(y_true, y_pred):
    return 1 - similarity(y_true, y_pred)

def dice_coef_mhw(y_true, y_pred):
    y_true_mhw = y_true[...,1]
    y_pred_mhw = y_pred[...,1]
    intersection_mhw = K.sum(y_true_mhw * y_pred_mhw)
    return (2 * intersection_mhw + smooth) / (K.sum(y_true_mhw) + K.sum(y_pred_mhw) + smooth)

def dice_coef_nn(y_true, y_pred):
    y_true_nn = y_true[...,0]
    y_pred_nn = y_pred[...,0]
    intersection_nn = K.sum(y_true_nn * y_pred_nn)
    return (2 * intersection_nn + smooth) / (K.sum(y_true_nn) + K.sum(y_pred_nn) + smooth)


def mean_dice_coef(y_true, y_pred):
    return (dice_coef_mhw(y_true, y_pred) + dice_coef_nn(y_true, y_pred)) / 2


def weighted_mean_dice_coef(y_true, y_pred):
    return (0.95 * dice_coef_mhw(y_true, y_pred) + 0.05 * dice_coef_nn(y_true, y_pred))

def dice_coef_loss(y_true, y_pred):
    return 1 - weighted_mean_dice_coef(y_true, y_pred)

#def my_loss(y_true, y_pred):

def global_accuracy(y_true, y_pred):
    true_positive = K.sum(y_true * y_pred)
    predicted_positive = K.sum(y_pred)
    global_accuracy = true_positive / predicted_positive
    return global_accuracy


def precision(y_true, y_pred):
    true_positive = K.sum(y_true[...,1] * y_pred[...,1])
    predicted_positive = K.sum(y_pred[...,1])
    precision = true_positive / predicted_positive
    return precision


def recall(y_true, y_pred):
    true_positive = K.sum(y_true[...,1] * y_pred[...,1])
    predicted_positive = K.sum(y_true[...,1])
    recall = true_positive / predicted_positive
    return recall

'''
def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('the lowest choosable beta is zero')
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    b = beta ** 2
    fbeta_score = (1 + b) * (p * r) / (b * p + r + K.epsilon())
    return fbeta_score
'''

def MCC(y_true, y_pred):
    TP = K.sum(y_true[...,1] * y_pred[...,1])
    TN = K.sum(y_true[...,0] * y_pred[...,0])
    FP = K.sum(y_true[...,0] * y_pred[...,1])
    FN = K.sum(y_true[...,1] * y_pred[...,0])
    MCC = (TP*TN-FN*FP)/K.sqrt((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN)+smooth)
    return MCC
def my_loss(y_true, y_pred):
    return 1-MCC

def SEDI(y_true, y_pred):
    TP = K.sum(y_true[...,1] * y_pred[...,1])
   # TN = K.sum(y_true[:,:,:,0] * y_pred[:,:,:,0])
    FP = K.sum(y_true[...,0] * y_pred[...,1])
   # FN = K.sum(y_true[:,:,:,1] * y_pred[:,:,:,0])
    H = TP / K.sum(y_true[...,1])
    F = FP / K.sum(y_true[...,0])
    SEDI = (K.log(F)-K.log(H)-K.log(1-F)+K.log(1-H))/(K.log(F)+K.log(H)+K.log(1-F)+K.log(1-H))
    return SEDI

mhwunet.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=['categorical_accuracy', mean_dice_coef,
                                                                          weighted_mean_dice_coef, 
                                                                          dice_coef_mhw, dice_coef_nn, precision,
                                                                          recall, #fbeta_score,
                                                                          MCC, SEDI, difference])

earl = EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=80, verbose=1, mode='auto')
modelcheck = ModelCheckpoint('./mhwunet_cnn_4_unit_lead14_mhw.h5', monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True)
reducecall = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', min_delta=1e-30,
                               min_lr=1e-30)
          
# Train / Validation split
available_ids = [i for i in range(0, t2_train.shape[3])]

#from random import shuffle
#shuffle(available_ids)

final_train_id = 2034#int(len(available_ids)*0.8)
train_ids = available_ids[:final_train_id]
val_ids = available_ids[final_train_id:]
print('final_train_id:')
print(final_train_id)
print('length 0f val:')
print(len(val_ids))

# import math
class DataGenerator(Sequence):
    def __init__(self,  sst_train, t2_train, Label_train_categor, batch_size, window_size, stride):
        self.sst_train, self.train, self.label = sst_train, t2_train, Label_train_categor
        self.batch_size = batch_size
        self.window_size = window_size
        self.stride = stride
        self.index = np.arange(self.train.shape[3])
        np.random.shuffle(self.index)

    def __len__(self):
        return int(np.ceil((self.train.shape[3]) / self.stride)) // self.batch_size

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for i in range(self.batch_size):
            start_idx = self.index[idx * self.batch_size * self.stride + i * self.stride]
          #  start_idx = idx * self.batch_size * self.stride + i * self.stride
            end_idx = start_idx + self.window_size
            a = self.sst_train[:,:,start_idx,:]  
           # b = np.concatenate((a,self.train[:,:,:,start_idx+1,0]),axis=2)
            b = np.concatenate((a,self.train[:,:,:,start_idx,0]),axis=2)
            batch_x.append(b)
           # batch_y.append(self.label[:,:,start_idx+1:end_idx+1,:])
            batch_y.append(self.label[:,:,:,start_idx,:])
        return np.array(batch_x), np.array(batch_y)

def Generator(DataGenerator):
    while True:
        for item in DataGenerator:
            yield item

batch_size = 4
window_size = 15
stride = 1

sequence_train = DataGenerator(sst_train[:,:,:final_train_id,:],t2_train[:,:,:,:final_train_id,:],Label_train_categor[:,:,:,:final_train_id,:],batch_size,window_size,stride)
generator_train = Generator(sequence_train)

sequence_val = DataGenerator(sst_train[:,:,final_train_id:,:],t2_train[:,:,:,final_train_id:,:],Label_train_categor[:,:,:,final_train_id:,:],batch_size,window_size,stride)
generator_val = Generator(sequence_val)

# fit the model
history = mhwunet.fit(
    generator_train
    , steps_per_epoch = int((len(train_ids))/batch_size)
    , validation_data = generator_val
    , validation_steps = int((len(val_ids))/batch_size)
    , epochs = 400
    , verbose = 1
    , shuffle = True
    , initial_epoch = 0
    , callbacks=[modelcheck, reducecall, earl]
    )

loss = mhwunet.history.history['loss']
val_loss = mhwunet.history.history['val_loss']
io.savemat('./loss_mhwunet.mat',{'loss': loss})
io.savemat('./val_loss_mhwunet.mat',{'val_loss': val_loss})

plt.figure(figsize=(13, 8))
#plt.semilogy(mhwunet.history.history['loss'])
#plt.semilogy(mhwunet.history.history['val_loss'])
plt.plot(mhwunet.history.history['loss'],linewidth=3)
plt.plot(mhwunet.history.history['val_loss'],linewidth=3)
plt.ylim(0.2, 0.5)
#plt.yticks(fontsize=18)
plt.yticks(np.arange(.2, .5, .1), fontsize = 16)
plt.xticks(np.arange(0,400,50),fontsize = 16)
plt.title('MHWUNet Loss', fontsize=25)
plt.ylabel('loss',fontsize=20)
plt.xlabel('epoch',fontsize=20)
plt.legend(['train_loss', 'test_loss'], loc='center right', fontsize=20)
plt.savefig('./mhwunet_cnn_4_unit_lead14_mhw.png')


preds = mhwunet.evaluate(generator_val,steps = int(len(val_ids)/4))

print('loss: %s,' % preds[0], 'accuracy: %s,' % preds[1], 'mean_dice_coef: %s,' % preds[2],
      'weighted_mean_dice_coef: %s,' % preds[3],
      'DiceCoef MHW: %s,' % preds[4], 'DiceCoef NN: %s' % preds[5],
      'precision: %s,' % preds[6], 'recall: %s,' % preds[7], 'MCC: %s' % preds[8], 'SEDI: %s' % preds[9], 'Diff: %s' % preds[10])

predictedSEGMimage = np.zeros((240,80,15,len(val_ids)))
for i in np.arange(0, len(val_ids)):
    tt = np.reshape(np.concatenate((sst_train[:,:,i+final_train_id,:],t2_train[:,:,:,i+final_train_id,0]),axis=2),(1,240,80,16))
    predictedSEGM = mhwunet.predict(tt)
    #    print(predictedSEGM.shape)
    predictedSEGMimage[...,i] = np.array(np.reshape(predictedSEGM.argmax(4),(240,80,15)),dtype='float32')

io.savemat('./predict_mhw.mat',{'mask': predictedSEGMimage})


'''
randindex=12370 #np.random.randint(0,SST_train_all.shape[0]-14)
predictedSEGM=mhwunet.predict(np.reshape(SST_train_all[randindex, :, :,:], (1, height, width, 14)))
print(predictedSEGM.shape)
predictedSEGMimage = np.reshape(predictedSEGM.argmax(3),(height,width))

plt.figure(figsize=(10, 15))

plt.subplot(311)
#plt.imshow(np.flip(SST_train[:,:,randindex+14,0].T), cmap='bwr')
#plt.colorbar(extend='both', fraction=0.042, pad=0.04)
#plt.clim(-0.25,0.25)
#plt.axis('off')
#plt.title('SSTA')
map = Basemap(
    projection='merc',
    llcrnrlon=0,
    llcrnrlat=-60,
    urcrnrlon=360,
    urcrnrlat=60,
    lat_0=-60.,lon_0=0.
    # resolution='i'
    )
csfont = {'fontname':'Times New Roman'}
lon = np.arange(0,360,1)
lat = np.arange(-60,60,1)
Lat, Lon = map(*np.meshgrid(lon, lat))
map.drawcoastlines(linewidth=0.5)
map.drawmapboundary(fill_color='w')
parallels = np.arange(-60, 60, 30)
meridians = np.arange(0, 360, 60)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=15)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=15)
map.fillcontinents(color='gray')
cs = map.pcolormesh(Lat, Lon, SST_train[:,:,randindex+14,0].T, vmin = -3, vmax = 3, cmap='seismic')
cb = map.colorbar(cs, extend='both',pad=0.13, size=0.15)
cb.set_label('℃',fontsize=15)
plt.title('SSTA' ,fontsize=20,**csfont)

plt.subplot(312)
#plt.imshow(np.flip(predictedSEGMimage.T), cmap='bwr')
#plt.colorbar(extend='both', fraction=0.042, pad=0.04)
#plt.clim(-0.25,0.25)
#plt.axis('off')
#plt.title('MHWUnet Segmentation')
map = Basemap(
    projection='merc',
    llcrnrlon=0,
    llcrnrlat=-60,
    urcrnrlon=360,
    urcrnrlat=60,
    lat_0=-60.,lon_0=0.
    # resolution='i'
    )
csfont = {'fontname':'Times New Roman'}
lon = np.arange(0,360,1)
lat = np.arange(-60,60,1)
Lat, Lon = map(*np.meshgrid(lon, lat))
map.drawcoastlines(linewidth=0.5)
map.drawmapboundary(fill_color='w')
parallels = np.arange(-60, 60, 30)
meridians = np.arange(0, 360, 60)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=15)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=15)
map.fillcontinents(color='gray')
cs = map.pcolormesh(Lat, Lon, predictedSEGMimage.T, vmin = -1, vmax = 1, cmap='seismic')
#cb = map.colorbar(cs, extend='both', pad=0.13, size=0.15)
#cb.set_label('℃',fontsize=15)
plt.title('MHWUnet Segmentation' ,fontsize=20,**csfont)

plt.subplot(313)
map = Basemap(
    projection='merc',
    llcrnrlon=0,
    llcrnrlat=-60,
    urcrnrlon=360,
    urcrnrlat=60,
    lat_0=-60.,lon_0=0.
    # resolution='i'
    )
csfont = {'fontname':'Times New Roman'}
lon = np.arange(0,360,1)
lat = np.arange(-60,60,1)
Lat, Lon = map(*np.meshgrid(lon, lat))
map.drawcoastlines(linewidth=0.5)
map.drawmapboundary(fill_color='w')
parallels = np.arange(-60, 60, 30)
meridians = np.arange(0, 360, 60)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=15)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=15)
map.fillcontinents(color='gray')
# x,y = map(Lon,Lat)
#a = SST_train[:,:,idx].T
cs = map.pcolormesh(Lat, Lon, Label_train[:,:,randindex+14,0].T, vmin = -1, vmax = 1, cmap='seismic')
plt.title('Ground Truth Segmentation',fontsize=20,**csfont)
plt.savefig('./test_show_filtered.png')
'''
