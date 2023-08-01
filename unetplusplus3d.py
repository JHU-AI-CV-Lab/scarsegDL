# Writing the code for Cunet3d.py

from __future__ import print_function 

import tensorflow as tf 
import numpy as np 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization, Add 
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

def standard_unit(xinput,n_convs,kernel) : 
    x = Conv3D(n_convs,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(xinput)
    x = BatchNormalization()(x)
    x = Conv3D(n_convs,kernel,activation='relu',padding='same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(x)
    x = BatchNormalization()(x)
    
    return x 


def get_unetpp3D_multi(img_rows, img_cols, slices, classes, layers, min_convs, kernel, learning_rate, loss, weights): 
    inputs = Input((img_rows,img_cols,slices,1))
    
    conv11 = standard_unit(inputs,min_convs,kernel)
    pool1 = MaxPooling3D(pool_size = (2,2,2))(conv11) # only layer where I pool in z 
    
    conv21 = standard_unit(pool1,min_convs*2,kernel)
    pool2 = MaxPooling3D(pool_size=(2,2,1))(conv21)
    
    up12 = Conv3DTranspose(min_convs,(2,2,2),strides=(2,2,2),padding="same",kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv21)
    conv12 = concatenate([up12,conv11],axis=4)
    conv12 = standard_unit(conv12,min_convs,kernel)
    
    conv31 = standard_unit(pool2,min_convs*4,kernel)
    pool3 = MaxPooling3D(pool_size = (2,2,1))(conv31)
    
    up22 = Conv3DTranspose(min_convs*2,(2,2,1),strides=(2,2,1),padding="same",kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv31)
    conv22 = concatenate([up22,conv21],axis=4)
    conv22 = standard_unit(conv22,min_convs*2,kernel)
    
    up13 = Conv3DTranspose(min_convs,(2,2,2),strides=(2,2,2),padding="same",kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv22)
    conv13 = concatenate([up13,conv11,conv12],axis=4)
    conv13 = standard_unit(conv13,min_convs,kernel)
    
    conv41 = standard_unit(pool3,min_convs*8,kernel)
    pool4 = MaxPooling3D(pool_size = (2,2,1))(conv41)
    
    up32 = Conv3DTranspose(min_convs*4,(2,2,1),strides=(2,2,1),padding="same",kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv41)
    conv32 = concatenate([up32,conv31],axis=4)
    conv32 = standard_unit(conv32,min_convs*4,kernel)
    
    up23 = Conv3DTranspose(min_convs*2,(2,2,1),strides=(2,2,1),padding="same",kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv32)
    conv23 = concatenate([up23,conv21,conv22],axis=4)
    conv23 = standard_unit(conv23,min_convs*2,kernel)
    
    up14 = Conv3DTranspose(min_convs,(2,2,2),strides=(2,2,2),padding="same",kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv23)
    conv14 = concatenate([up14,conv11,conv12,conv13],axis=4)
    conv14 = standard_unit(conv14,min_convs,kernel)
    
    conv51 = standard_unit(pool4,min_convs*16,kernel)
    
    up42 = Conv3DTranspose(min_convs*8,(2,2,1),strides=(2,2,1),padding="same",kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv51)
    conv42 = concatenate([up42,conv41],axis=4)
    conv42 = standard_unit(conv42,min_convs*8,kernel)
    
    up33 = Conv3DTranspose(min_convs*4,(2,2,1),strides=(2,2,1),padding="same",kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv42)
    conv33 = concatenate([up33,conv31,conv32],axis=4)
    conv33 = standard_unit(conv33,min_convs*4,kernel)
    
    up24 = Conv3DTranspose(min_convs*2,(2,2,1),strides=(2,2,1),padding="same",kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv33)
    conv24 = concatenate([up24,conv21,conv22,conv23],axis=4)
    conv24 = standard_unit(conv24,min_convs*2,kernel)
    
    up15 = Conv3DTranspose(min_convs,(2,2,2),strides=(2,2,2),padding="same",kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv24)
    conv15 = concatenate([up15,conv11,conv12,conv13,conv14],axis=4)
    conv15 = standard_unit(conv15,min_convs,kernel)
    
    
    out1 = Conv3D(classes,(1,1,1),activation = 'softmax',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv12)
    out2 = Conv3D(classes,(1,1,1),activation = 'softmax',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv13)
    out3 = Conv3D(classes,(1,1,1),activation = 'softmax',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv14)
    out4 = Conv3D(classes,(1,1,1),activation = 'softmax',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv15)
    
    outputs = Add()([out1,out2,out3,out4])
    
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




