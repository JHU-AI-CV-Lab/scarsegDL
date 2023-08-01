# Writing the code for Cunet3d.py

from __future__ import print_function 

import tensorflow as tf 
import numpy as np 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization, Add, SpatialDropout3D
from tensorflow.keras import regularizers 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import backend as K
from tensorflow.keras.losses import KLDivergence

# Dice Fuction (Accuracy Metric) 
def dice_coef(y_true,y_pred):
    y_true = K.argmax(y_true,axis = -1) 
    y_true = K.cast(y_true,'int32')
    
    y_pred = K.argmax(y_pred, axis = -1) 
    y_pred = K.cast(y_pred,'int32') 
    
    y_true_f = K.flatten(y_true) 
    y_pred_f = K.flatten(y_pred) 
    
    yand = K.all(K.stack([y_true_f, y_pred_f], axis=0), axis=0)
    yand = K.cast(yand, 'float32') 
    intersection = K.sum(yand) 
    
    # Determine Cardinalities 
    y_true_clip = K.cast(K.clip(y_true,0,1),'float32') 
    Y1 = K.sum(y_true_clip)
    y_pred_clip = K.cast(K.clip(y_pred,0,1),'float32') 
    Y2 = K.sum(y_pred_clip) 
    
    return (2*intersection) / (Y1 + Y2) 

def myo_dice(y_true,y_pred):
    y_true = K.argmax(y_true,axis = -1) 
    y_true = K.cast(y_true,'int32')
    
    y_pred = K.argmax(y_pred, axis = -1) 
    y_pred = K.cast(y_pred,'int32') 
    
    y_true_f = K.flatten(y_true) 
    y_pred_f = K.flatten(y_pred) 
    
    y_true_f = K.clip(y_true_f,0,1)
    y_pred_f = K.clip(y_pred_f,0,1)
    
    yand = K.all(K.stack([y_true_f, y_pred_f], axis=0), axis=0)
    yand = K.cast(yand, 'float32') 
    intersection = K.sum(yand) 
    
    # Determine Cardinalities 
    y_true_clip = K.cast(K.clip(y_true,0,1),'float32') 
    Y1 = K.sum(y_true_clip)
    y_pred_clip = K.cast(K.clip(y_pred,0,1),'float32') 
    Y2 = K.sum(y_pred_clip) 

    
    return (2*intersection) / (Y1 + Y2) 


def scar_percent_ratio(y_true,y_pred): 
    y_true_f = K.argmax(y_true,axis = -1) 
    y_true_f = K.flatten(y_true_f)
    y_pred_f = K.argmax(y_pred,axis = -1) 
    y_pred_f = K.flatten(y_pred_f)
    
    y_true_scar = K.equal(y_true_f,2)
    y_true_scar = K.cast(y_true_scar,'float32')
    y_true_scar = K.sum(y_true_scar)
    y_true_myo = K.equal(y_true_f,1)
    y_true_myo = K.cast(y_true_myo,'float32')
    y_true_myo = K.sum(y_true_myo)
    scar_per_true = y_true_scar/(y_true_scar + y_true_myo)
    
    y_pred_scar = K.equal(y_pred_f,2)
    y_pred_scar = K.cast(y_pred_scar,'float32')
    y_pred_scar = K.sum(y_pred_scar),2
    y_pred_myo = K.equal(y_pred_f,1)
    y_pred_myo = K.cast(y_pred_myo,'float32')
    y_pred_myo = K.sum(y_pred_myo)
    scar_per_pred = y_pred_scar/(y_pred_scar + y_pred_myo)
    
    
    out = scar_per_pred / scar_per_true
    
    return out

# Loss Functions 

def dice_loss(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_val = (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

    return (1-dice_val)

# Weighted categorical cross entropy 
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


def dice_gen_loss(weights): 
    
    weights = K.variable(weights) 
    def loss(y_true,y_pred): 
        
        # scale predictions so that the class probas of each sample sum to 1
        y_pred1 = K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred1 = K.clip(y_pred1, K.epsilon(), 1 - K.epsilon())
        # calc
        loss1 = y_true * K.log(y_pred1) * weights
        loss1 = -K.sum(loss1, -1)
        
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        y_pred_f = K.clip(y_pred_f, K.epsilon(), 1 - K.epsilon())
        intersection = K.sum(y_true_f * y_pred_f)
        dice_val = (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
        
        loss = loss1 + (1 - dice_val) 
        return loss 
    
    return loss 


def wcce_kld(weights):
    weights = K.variable(weights)
    
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss1 = y_true * K.log(y_pred) * weights
        loss1 = -K.sum(loss1, -1)
        kl = KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
        loss2 = kl(y_true, y_pred)
        loss = loss1 + loss2 
        return loss
    
    return loss 
    


def get_Cunet3D_mulit(img_rows, img_cols, slices, classes, layers, min_convs, kernel, learning_rate, loss, weights): 
    inputs = Input((img_rows,img_cols,slices,1))
    
    # U Net 1 
    conv11 = Conv3D(min_convs,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(inputs)
    conv11 = BatchNormalization()(conv11)
    conv11 = SpatialDropout3D(0.01)(conv11)  
    conv11 = Conv3D(min_convs,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv11)
    conv11 = BatchNormalization()(conv11)
    conv11 = SpatialDropout3D(0.01)(conv11)  
    pool11 = MaxPooling3D(pool_size = (2,2,2))(conv11) # This is the only layer where I pool in 3D 
    
    conv12 = Conv3D(min_convs*2,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(pool11)
    conv12 = BatchNormalization()(conv12)
    conv12 = SpatialDropout3D(0.01)(conv12)  
    conv12 = Conv3D(min_convs*2,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv12)
    conv12 = BatchNormalization()(conv12)
    conv12 = SpatialDropout3D(0.01)(conv12)  
    pool12 = MaxPooling3D(pool_size = (2,2,1))(conv12) # no pooling in z 
    
    conv13 = Conv3D(min_convs*4,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(pool12)
    conv13 = BatchNormalization()(conv13)
    conv13 = SpatialDropout3D(0.01)(conv13)  
    conv13 = Conv3D(min_convs*4,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv13)
    conv13 = BatchNormalization()(conv13)
    conv13 = SpatialDropout3D(0.01)(conv13) 
    pool13 = MaxPooling3D(pool_size = (2,2,1))(conv13) # no pooling in z 
    
    conv14 = Conv3D(min_convs*8,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(pool13)
    conv14 = BatchNormalization()(conv14)
    conv14 = SpatialDropout3D(0.01)(conv14) 
    conv14 = Conv3D(min_convs*8,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv14)
    conv14 = BatchNormalization()(conv14)
    conv14 = SpatialDropout3D(0.01)(conv14)
    pool14 = MaxPooling3D(pool_size = (2,2,1))(conv14) # no pooling in z 
    
    # U Net 1 - Bottom
    conv15 = Conv3D(min_convs*16,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(pool14)
    conv15 = BatchNormalization()(conv15)
    conv15 = SpatialDropout3D(0.01)(conv15)
    conv15 = Conv3D(min_convs*16,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv15)
    conv15 = BatchNormalization()(conv15)
    conv15 = SpatialDropout3D(0.01)(conv15)
    
    # U Net 1 - Ascending limb 
    up16 = Conv3DTranspose(min_convs*8,(2,2,1),strides=(2,2,1),padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv15)
    up16 = concatenate([up16,conv14],axis=4)
    conv16 = Conv3D(min_convs*8,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(up16)
    conv16 = BatchNormalization()(conv16)
    conv16 = SpatialDropout3D(0.01)(conv16)
    conv16 = Conv3D(min_convs*8,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv16)
    conv16 = BatchNormalization()(conv16)
    conv16 = SpatialDropout3D(0.01)(conv16)
    
    up17 = Conv3DTranspose(min_convs*4,(2,2,1),strides=(2,2,1),padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv16)
    up17 = concatenate([up17,conv13],axis=4)
    conv17 = Conv3D(min_convs*4,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(up17)
    conv17 = BatchNormalization()(conv17)
    conv17 = SpatialDropout3D(0.01)(conv17)
    conv17 = Conv3D(min_convs*4,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv17)
    conv17 = BatchNormalization()(conv17)
    conv17 = SpatialDropout3D(0.01)(conv17)
    
    up18 = Conv3DTranspose(min_convs*2,(2,2,1),strides=(2,2,1),padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv17)
    up18 = concatenate([up18,conv12],axis=4)
    conv18 = Conv3D(min_convs*2,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(up18)
    conv18 = BatchNormalization()(conv18)
    conv18 = SpatialDropout3D(0.01)(conv18)
    conv18 = Conv3D(min_convs*2,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv18)
    conv18 = BatchNormalization()(conv18)
    conv18 = SpatialDropout3D(0.01)(conv18)
    
    up19 = Conv3DTranspose(min_convs,(2,2,2),strides=(2,2,2),padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv18)
    up19 = concatenate([up19,conv11],axis=4)
    conv19 = Conv3D(min_convs,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(up19)
    conv19 = BatchNormalization()(conv19)
    conv19 = SpatialDropout3D(0.01)(conv19)
    conv19 = Conv3D(min_convs,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv19)
    out1 = BatchNormalization()(conv19)
    conv19 = SpatialDropout3D(0.01)(conv19)
    
    # U Net 2 
    pool21 = MaxPooling3D(pool_size = (2,2,2))(out1) # This is the only layer where I pool in 3D
    
    conv22 = concatenate([pool21,conv18],axis = 4)
    conv22 = Conv3D(min_convs*2,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv22)
    conv22 = BatchNormalization()(conv22)
    conv22 = SpatialDropout3D(0.01)(conv22)
    conv22 = Conv3D(min_convs*2,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv22)
    conv22 = BatchNormalization()(conv22)
    conv22 = SpatialDropout3D(0.01)(conv22)
    pool22 = MaxPooling3D(pool_size = (2,2,1))(conv22) 
    
    conv23 = concatenate([pool22,conv17])
    conv23 = Conv3D(min_convs*4,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv23)
    conv23 = BatchNormalization()(conv23)
    conv23 = SpatialDropout3D(0.01)(conv23)
    conv23 = Conv3D(min_convs*4,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv23)
    conv23 = BatchNormalization()(conv23)
    conv23 = SpatialDropout3D(0.01)(conv23)
    pool23 = MaxPooling3D(pool_size=(2,2,1))(conv23)
    
    conv24 = concatenate([pool23,conv16])
    conv24 = Conv3D(min_convs*8,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv24)
    conv24 = BatchNormalization()(conv24)
    conv24 = SpatialDropout3D(0.01)(conv24)
    conv24 = Conv3D(min_convs*8,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv24)
    conv24 = BatchNormalization()(conv24)
    conv24 = SpatialDropout3D(0.01)(conv24)
    pool24 = MaxPooling3D(pool_size=(2,2,1))(conv24)
    
    conv25 = concatenate([pool24,conv15])
    conv25 = Conv3D(min_convs*16,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv25)
    conv25 = BatchNormalization()(conv25)
    conv25 = SpatialDropout3D(0.01)(conv25)
    conv25 = Conv3D(min_convs*16,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv25)
    conv25 = BatchNormalization()(conv25)
    conv25 = SpatialDropout3D(0.01)(conv25)

    up26 = Conv3DTranspose(min_convs*8,(2,2,1),strides=(2,2,1),padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv25)
    up26 = concatenate([up26,conv24],axis=4)
    conv26 = Conv3D(min_convs*8,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(up26)
    conv26 = BatchNormalization()(conv26)
    conv26 = SpatialDropout3D(0.01)(conv26)
    conv26 = Conv3D(min_convs*8,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv26)
    conv26 = BatchNormalization()(conv26)
    conv26 = SpatialDropout3D(0.01)(conv26)
    
    up27 = Conv3DTranspose(min_convs*4,(2,2,1),strides=(2,2,1),padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv26)
    up27 = concatenate([up27,conv23],axis=4)
    conv27 = Conv3D(min_convs*4,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(up27)
    conv27 = BatchNormalization()(conv27)
    conv27 = SpatialDropout3D(0.01)(conv27)
    conv27 = Conv3D(min_convs*4,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv27)
    conv27 = BatchNormalization()(conv27)
    conv27 = SpatialDropout3D(0.01)(conv27)
    
    up28 = Conv3DTranspose(min_convs*2,(2,2,1),strides=(2,2,1),padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv27)
    up28 = concatenate([up28,conv22],axis=4)
    conv28 = Conv3D(min_convs*2,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(up28)
    conv28 = BatchNormalization()(conv28)
    conv28 = SpatialDropout3D(0.01)(conv28)
    conv28 = Conv3D(min_convs*2,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv28)
    conv28 = BatchNormalization()(conv28)
    conv28 = SpatialDropout3D(0.01)(conv28)
    
    up29 = Conv3DTranspose(min_convs,(2,2,2),strides=(2,2,2),padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv28)
    up29 = concatenate([up29,out1],axis=4)
    conv29 = Conv3D(min_convs,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(up29)
    conv29 = BatchNormalization()(conv29)
    conv29 = SpatialDropout3D(0.01)(conv29)
    conv29 = Conv3D(min_convs,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv29)
    out2 = BatchNormalization()(conv29)
    conv29 = SpatialDropout3D(0.01)(conv29)
    
    out12 = Add()([out1,out2])
    outputs = Conv3D(classes,(1,1,1),activation = 'softmax')(out12)
    
    model = Model(inputs = [inputs], outputs = [outputs])
    
    if (loss == 'dice_loss'):
        model.compile(optimizer = Adam(lr = learning_rate), loss = dice_loss, metrics =['accuracy',dice_coef,myo_dice])
    elif (loss == 'weighted_categorical_crossentropy'):
        ncce = weighted_categorical_crossentropy(weights)
        model.compile(optimizer = Adam(lr = learning_rate),loss = ncce, metrics = ['accuracy',dice_coef,myo_dice])
    elif (loss == 'dice_gen_loss'): 
        ncce = dice_gen_loss(weights) 
        model.compile(optimizer = Adam(lr = learning_rate),loss = ncce, metrics = ['accuracy',dice_coef,myo_dice])
    elif (loss == "wcce_kld"): 
        ncce = wcce_kld(weights) 
        model.compile(optimizer = Adam(lr = learning_rate),loss = ncce, metrics = ['accuracy',dice_coef,myo_dice])
    else:
        model.compile(optimizer = Adam(lr = learning_rate),loss = 'categorical_crossentropy', metrics=['accuracy',dice_coef,myo_dice])
    return model 

