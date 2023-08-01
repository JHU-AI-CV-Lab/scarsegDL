# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:45:17 2018

@author: vjani1
"""

# Imports 
import tensorflow as tf
from tensorflow import keras 
import os 
from tensorflow.keras.models import Model 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np 
from skimage.io import imsave 
import cv2 
from scipy import ndimage 
import random 
import tensorflow_addons as tfa 


@tf.function 
def augment(vol_lab_tuple):
    def tf_rotate(vol_lab_tuple):
        volume = vol_lab_tuple[:,:,:,0]
        labels = vol_lab_tuple[:,:,:,1]
        
        rows, cols, slices = volume.shape 
        angles_all = np.linspace(-20,20,1000)
        angle = random.choice(angles_all)
        
        volume = tfa.image.rotate(volume,angle)
        labels = tfa.image.rotate(labels,angle)
        
        volume[volume < -1] = -1
        volume[volume > 1] = 1 
        
        labels = tf.round(labels)
        labels[labels < 0] = 0
        labels[labels > 2] = 2
        
        new_out = tf.stack((volume,labels),-1)
        return new_out
    
    new_out = tf.numpy_function(tf_rotate,[vol_lab_tuple], tf.float32)
    
    return new_out 

def train_preprocessing(volume,label):
    
    volume.set_shape([256,256,16]) # THIS IS A TENSORFLOW BUG AND WILL NEED TO BE CHANGED MANUALLY 
    label.set_shape([256,256,16]) # THIS IS A TENSORFLOW BUG AND WILL NEED TO BE CHANGED MANUALLY
    
    #grouped_dat = np.ndarray(shape=(volume.shape[0],volume.shape[1],volume.shape[2],2),dtype=np.float32)
    label = tf.cast(label,tf.float32)
    grouped_dat = tf.stack((volume,label),-1) 
    grouped_dat.set_shape([256,256,16,2])
    
    
    grouped_out = augment(grouped_dat)
    volume2 = grouped_out[:,:,:,0]
    label2 = grouped_out[:,:,:,1]
    
    s = np.random.uniform(0,1)
    
    if s > 0.5: 
        volume2 = tf.flip(volume2,axis=2)
        label2 = tf.flip(label2,axis=2) 
    
    volume2 = tf.expand_dims(volume2,axis = 3)
    #label = tf.expand_dims(volume,axis = 3)
    label2 = tf.cast(label2,tf.int32)
    label_cat = tf.one_hot(label2, 3)
    
    volume2.set_shape([256,256,16,1]) # THIS IS A TENSORFLOW BUG AND WILL NEED TO BE CHANGED MANUALLY 
    label_cat.set_shape([256,256,16,3]) # THIS IS A TENSORFLOW BUG AND WILL NEED TO BE CHANGED MANUALLY
    
    return volume, label_cat 

def validation_preprocessing(volume,label): 
    volume = tf.expand_dims(volume, axis = 3)
    #label = tf.expand_dims(volume, axis = 3)
    label = tf.cast(label,tf.int32)
    label_cat = tf.one_hot(label, 3)
    
    volume.set_shape([256,256,16,1]) # THIS IS A TENSORFLOW BUG AND WILL NEED TO BE CHANGED MANUALLY
    label_cat.set_shape([256,256,16,3]) # THIS IS A TENSORFLOW BUG AND WILL NEED TO BE CHANGED MANUALLY
    
    return volume, label_cat 

def training_main(imgs_train,masks_train,model,savepath, batches, epochs, imgs_val,masks_val):    
    print('-'*50)
    print('Fitting Model...')
    print('-'*50)
    
    train_loader = tf.data.Dataset.from_tensor_slices((imgs_train,masks_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((imgs_val,masks_val))
    
    train_dataset = (
        train_loader.shuffle(len(imgs_train))
        .map(validation_preprocessing)
        .batch(batches,drop_remainder=True)
        .prefetch(batches)
        .repeat()
    )
    
    validation_dataset = (
        validation_loader.shuffle(len(imgs_val))
        .map(validation_preprocessing)
        .batch(batches,drop_remainder=True)
        .prefetch(batches)
        .repeat()
    )
    
    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_dataset = train_dataset.with_options(options)
    validation_dataset = validation_dataset.with_options(options)
    
    # Define callbacks.
    weight_name = savepath+'final_weights.h5'
    
    checkpoint_cb = keras.callbacks.ModelCheckpoint(weight_name, save_best_only=True)
        
    history = model.fit(train_dataset, epochs = epochs, 
              verbose = 1, shuffle = True, steps_per_epoch=5000, validation_data = validation_dataset, 
                       callbacks = [checkpoint_cb],validation_steps = 35)
 
    
    print('-'*50)
    print('Saving Model Weights (Load Later into SAME Model Architecture)...')
    print('-'*50)
    
    
    #model.save_weights(weight_name)
    
    # Saving training and validation history 
    # train_loss = history.history['loss']
    # train_loss = np.array(train_loss)
    # np.save(savepath+'train_loss.npy',train_loss)
    
    # train_dice = history.history['dice_coef']
    # train_dice = np.array(train_dice)
    # np.save(savepath+'train_dice.npy',train_dice)
    
    # val_loss = history.history['val_loss']
    # val_loss = np.array(val_loss)
    # np.save(savepath+'val_loss.npy',val_loss)
    
    # val_dice = history.history['val_dice_coef']
    # val_dice = np.array(val_dice)
    # np.save(savepath+'val_dice.npy',val_dice)
    
    
    
    
    return history  
