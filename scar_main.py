# Required Libraries 
import numpy as np 
import os 

# Keras Libraries 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


# My libraries 
from training import training_main
from unet3d import get_unet3D_multi 
from cunet3d import get_Cunet3D_mulit
from training import training_main 
from testing3D import testing_parallel
from unetplusplus3d import get_unetpp3D_multi


print('-'*100)
print('Loading Data...')
print('-'*100)

dpath = '/home/vivek/scar_data2/' 

shift = np.load(dpath+'scar_mean.npy')

imgs_train = np.load(dpath+'scar_imgs_train.npy')
imgs_val = np.load(dpath+'scar_imgs_test.npy')
segs_train = np.load(dpath+'scar_segs_train.npy')
segs_val = np.load(dpath+'scar_segs_test.npy')

# Define new path for saving data
# General naming format is 'home/vivek/scar_model_kernel_minconvs_weights_batch_epoch_lrate_lossfn
path1 = '/home/vivek/scar_3DCUnetreg01drop001_333_4_1201000_4_100_1e-4_awccekld098_newtrain/' # NAME 

# Create the path 
if not os.path.exists(path1):
    os.mkdir(path1)

print('-'*100)
print('Defining Model...')
print('-'*100)

# Image Parameters 
img_rows = 256
img_cols = 256
slices = 16 


# Model Parameters 
classes = 3 
layers = 4
min_convs = 4
kernel = (3,3,3)
lossfun = 'wcce_kld' # Need to write something for this later - might clean up model code as well 



# Hyperparameters 
decay = 0.98
learning_rate = 1e-4
batches = 4
epochs_total = 100
epochs_batch = 5 
weights_init = (1,20,1000) 


strategy = tf.distribute.MirroredStrategy()

train_loss = np.zeros((epochs_total,))
train_dice = np.zeros((epochs_total,))
train_myo = np.zeros((epochs_total,))

val_loss = np.zeros((epochs_total,))
val_dice = np.zeros((epochs_total,))
val_myo = np.zeros((epochs_total,))

with strategy.scope():
    model = get_Cunet3D_mulit(img_rows, img_cols, slices, classes, layers, min_convs, kernel, learning_rate, lossfun, weights_init)
model.summary()

cache = training_main(imgs_train,segs_train,model,path1, batches, epochs_batch, imgs_val, segs_val)
num_epoch_batches = int(np.round(epochs_total/epochs_batch))

train_loss[0:epochs_batch] = np.array(cache.history['loss'])
train_dice[0:epochs_batch] = np.array(cache.history['dice_coef'])
train_myo[0:epochs_batch] = np.array(cache.history['myo_dice'])

val_loss[0:epochs_batch] = np.array(cache.history['val_loss'])
val_dice[0:epochs_batch] = np.array(cache.history['val_dice_coef'])
val_myo[0:epochs_batch] = np.array(cache.history['val_myo_dice'])

np.save(path1+'train_loss.npy',train_loss)
np.save(path1+'train_dice.npy',train_dice)
np.save(path1+'train_myo.npy',train_myo)

np.save(path1+'val_loss.npy',val_loss)
np.save(path1+'val_dice.npy',val_dice)
np.save(path1+'val_myo.npy',val_myo)

for ii in range(num_epoch_batches - 1): 
    weights1 = (weights_init[1]-1)*np.exp(-ii*decay) + 1 
    weights2= (weights_init[2]-1)*np.exp(-ii*decay) + 1
    
    weights = (weights_init[0],weights1,weights2)

    with strategy.scope():
        model = get_Cunet3D_mulit(img_rows, img_cols, slices, classes, layers, min_convs, kernel, learning_rate, lossfun, weights)

    old_model_weight_path = path1 + 'final_weights.h5'
    model.load_weights(old_model_weight_path)
    
    ind1 = epochs_batch*(ii+1)
    ind2 = epochs_batch*(ii+2)
    
    print('-'*100)
    print('Training Epochs'+str(ind1)+"-"+str(ind2))
    print('-'*100)
    cache = training_main(imgs_train,segs_train,model,path1, batches, epochs_batch, imgs_val, segs_val)
    
    
    train_loss[ind1:ind2] = np.array(cache.history['loss'])
    train_dice[ind1:ind2] = np.array(cache.history['dice_coef'])
    train_myo[ind1:ind2] = np.array(cache.history['myo_dice'])

    val_loss[ind1:ind2] = np.array(cache.history['val_loss'])
    val_dice[ind1:ind2] = np.array(cache.history['val_dice_coef'])
    val_myo[ind1:ind2] = np.array(cache.history['val_myo_dice'])
    
    np.save(path1+'train_loss.npy',train_loss)
    np.save(path1+'train_dice.npy',train_dice)
    np.save(path1+'train_myo.npy',train_myo)

    np.save(path1+'val_loss.npy',val_loss)
    np.save(path1+'val_dice.npy',val_dice)
    np.save(path1+'val_myo.npy',val_myo)
    

np.save(path1+'train_loss.npy',train_loss)
np.save(path1+'train_dice.npy',train_dice)
np.save(path1+'train_myo.npy',train_myo)

np.save(path1+'val_loss.npy',val_loss)
np.save(path1+'val_dice.npy',val_dice)
np.save(path1+'val_myo.npy',val_myo)

print('-'*100)
print('Validating Model...')
print('-'*100)

savepath = path1+'validation/'

# Create the path 
if not os.path.exists(savepath):
    os.mkdir(savepath)

gpus = 2 

# Testing model on validation set only 
preds, contours = testing_parallel(savepath, model, imgs_val, segs_val, gpus, classes, shift)

