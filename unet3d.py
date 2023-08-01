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
    


def get_unet3D_multi(img_rows, img_cols, slices, classes, layers, min_convs, kernel, learning_rate, loss, weights): 
    
    inputs = Input((img_rows,img_cols,slices,1))
    encoding_convs = [] 
    
    for ii in range(layers): 
        N_convs = min_convs*(2**ii)
        
        if(ii == 0):
            conv0 = Conv3D(N_convs,kernel,activation = 'relu', padding = 'same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(inputs)
            conv0 = BatchNormalization()(conv0)
            conv0 = SpatialDropout3D(0.01)(conv0)
        else:
            conv0 = Conv3D(N_convs,kernel,activation = 'relu', padding = 'same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(pool)
            conv0 = BatchNormalization()(conv0)
            conv0 = SpatialDropout3D(0.01)(conv0)
        
        
        conv = Conv3D(N_convs,kernel,activation = 'relu', padding = 'same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv0)
        conv = BatchNormalization()(conv)
        encoding_convs.append(conv)
        
        pool = MaxPooling3D(pool_size = (2,2,1))(conv) 
        pool = SpatialDropout3D(0.01)(pool)    
    
    # Bottom Layer 
    conv2 = Conv3D(N_convs*2, kernel, activation = 'relu', padding = 'same',kernel_regularizer =tf.keras.regularizers.l2( l=0.01))(pool)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(N_convs*2, kernel, activation = 'relu', padding = 'same',kernel_regularizer =tf.keras.regularizers.l2( l=0.01))(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = SpatialDropout3D(0.01)(conv2)  
    
    for jj in range(layers): 
        N_convs_down = min_convs*(2**(layers - jj - 1))
        
        if jj == 0:
            up = Conv3DTranspose(N_convs_down, (2,2,1), strides = (2,2,1), padding = 'same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv2)
            up = BatchNormalization()(up)
            up = SpatialDropout3D(0.01)(up)
        else: 
            up = Conv3DTranspose(N_convs_down, (2,2,1), strides = (2,2,1), padding = 'same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv3)
            up = BatchNormalization()(up)
            up = SpatialDropout3D(0.01)(up)
           
        up = concatenate([up,encoding_convs[layers-1-jj]], axis = 4)
        conv3 = Conv3D(N_convs_down,kernel,activation = 'relu', padding = 'same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(up)
        conv3 = BatchNormalization()(conv3)
        conv3 = SpatialDropout3D(0.01)(conv3)
        conv3 = Conv3D(N_convs_down,kernel,activation = 'relu', padding = 'same',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = SpatialDropout3D(0.01)(conv3)
    outputs = Conv3D(classes,(1,1,1),activation = 'softmax',kernel_regularizer =tf.keras.regularizers.l2( l=0.1))(conv3)
    
    model = Model(inputs = [inputs], outputs = [outputs])
    
    # Compile model based on loss function that is chosen 
    
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
        
        

           





