import os
import cv2
import glob
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler

sm.set_framework('tf.keras')

sm.framework()

BACKBONE_3 = 'resnet152'
preprocessing_input = sm.get_preprocessing(BACKBONE_3)

seed=24
batch_size= 4
n_classes=9

# data normalization
scaler = MinMaxScaler()

def data_preprocessing(img, mask, num_class):
      
    #Scale images  
    #img = img[0:512,0:512]
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)   # fit and transform image using MinMaxScaler
    
    # preprocessing the input if a backbone is used else comment the line below if you want to use just UNET
    img = preprocessing_input(img)
    #mask = mask[0:512,0:512]
    # label encoding for the mask image
    labelencoder = LabelEncoder()                                                   # initializing Labelencoder
    number_of_images, height, width, channles= mask.shape                           # shape of the mask image
    mask_reshape = mask.reshape(-1,1)                                               # reshaping the mask image numpy array
    encoded_mask = labelencoder.fit_transform(mask_reshape.ravel())                 # fit and transform mask image using label encoder
    original_encoded_mask = encoded_mask.reshape(number_of_images, height, width )   # reshaping the image numpy array
    mask = np.expand_dims(original_encoded_mask, axis = 3)                          # expanding dimension (requirement by the model)

    #Convert mask to one-hot encoding
    mask = to_categorical(mask, num_class)                                          # into to categorical pixel values

    return (img,mask)

    def TFDataLoader(train_img_path, train_mask_path, num_class):
    
    # augmention parameters for images
    img_data_gen_args = dict(
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest'
                      )

    # initializing ImageDataGenerator for both images and masks
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)
    
    # images will be loaded directly from the local drive (less load on the memory)
    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        target_size=(256, 256),  
        class_mode = None,
        color_mode = "rgb",
        batch_size = batch_size,
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        target_size=(256, 256),
        class_mode = None,
        color_mode = 'grayscale',
        batch_size = batch_size,
        seed = seed)
        
    # zip both images and mask 
    data_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in data_generator:
        img, mask = data_preprocessing(img, mask, num_class)
        yield (img, mask)
# path for both train and val datasets     
train_img_path = "../../data/unet_img/Data_TF_Split/train_image/"                    
train_mask_path = "../../data/unet_img/Data_TF_Split/train_mask/"
train_img_gen = TFDataLoader(train_img_path, train_mask_path, num_class=9)          # calling TFDataLoader for training datasets  

val_img_path = "../../data/unet_img/Data_TF_Split/val_image/"
val_mask_path = "../../data/unet_img/Data_TF_Split/val_mask/"
val_img_gen = TFDataLoader(val_img_path, val_mask_path, num_class=9)  


test_img_path = "../../data/unet_img/Data_TF_Split/test_image/"
test_mask_path = "../../data/unet_img/Data_TF_Split/test_mask/"
test_img_gen = TFDataLoader(test_img_path, test_mask_path, num_class=9)
x_train, y_train = train_img_gen.__next__() # data iterator

# checking/ verifying if the image and masks are coorelated
for i in range(0,3):
    image = x_train[i]
    mask = np.argmax(y_train[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap = 'gray' ) 
    plt.show()

x_val, y_val = val_img_gen.__next__()
for i in range(0,3):
    image = x_val[i]
    mask = np.argmax(y_val[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.show()


IMG_HEIGHT = x_train.shape[1]
IMG_WIDTH  = x_train.shape[2]
IMG_CHANNELS = x_train.shape[3]
print(IMG_CHANNELS)

# loss function:
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1*focal_loss)

metrics = [sm.metrics.IOUScore(threshold = 0.5), sm.metrics.FScore(threshold = 0.5),'accuracy']

keras.backend.clear_session()

train_img_path_len = "../../data/unet_img/Data_TF_Split/train_image/JPEGImages"
img_list_len = len(os.listdir(train_img_path_len))
print(img_list_len)

# inout for the model
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
# model intializing (using UNET from Segmentation model lib)
model_3 = sm.Unet(BACKBONE_3, 
                encoder_weights = 'imagenet', 
                input_shape = input_shape, 
                classes = n_classes,
                activation = 'softmax')
learning_rate = 1e-4
model_3.compile(optimizer = Adam(learning_rate = learning_rate),
                loss = total_loss, 
                metrics = metrics)

#Step 3: Initialize Tensorboard to monitor changes in Model Loss 
import datetime
%load_ext tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#Visualize on tensorboard (move this above)
%tensorboard --logdir logs/fit

%reload_ext tensorboard

model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_resnet152_300ep.hdf5', monitor='loss',verbose=1)

steps_per_epoch = img_list_len //batch_size
history_3 = model_3.fit(train_img_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs = 300,
                    validation_data=val_img_gen, 
                    validation_steps=steps_per_epoch,
                    verbose = 1, callbacks = [model_checkpoint, tensorboard_callback] )

train_IoU = model_3.evaluate(train_img_gen,
                                batch_size = batch_size,
                                steps = steps_per_epoch)
print("Train IoU is = ", (train_IoU[1] * 100.0), "%")

val_IoU = model_3.evaluate(val_img_gen,
                                batch_size = batch_size,
                                steps = steps_per_epoch)
print("Val IoU is = ", (val_IoU[1] * 100.0), "%")

model_3.save('unet_resnet152_300ep.hdf5.hdf5')

#plot the training and validation IoU and loss at each epoch
loss = history_3.history['loss']
val_loss = history_3.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


iou_train = history_3.history['iou_score']
iou_val = history_3.history['val_iou_score']

plt.plot(epochs, iou_train, 'y', label='Training IoU')
plt.plot(epochs, iou_val, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()

from keras.models import load_model
model_effnet = load_model('unet_resnet152_300ep.hdf5', compile=False)

test_image_batch, test_mask_batch = test_img_gen.__next__()

#Convert categorical to integer for visualization and IoU calculation
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3) 
test_pred_batch = model_effnet.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)

n_classes = 9
IOU_keras_mod3 = MeanIoU(num_classes=n_classes)  
IOU_keras_mod3.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras_mod3.result().numpy())

import pandas as pd

weight_values_3 = np.array(IOU_keras_mod3.get_weights()).reshape(n_classes, n_classes)
classes = ['Background','Wings','Proboscis','Head','Antennae','Palps','Abdomen','Legs','Thorax']
weight_val_df = pd.DataFrame(weight_values_3, index = classes, columns= classes)

#print(weight_values)
#ax = plt.figure(figsize=(10, 10))
#ax = sns.heatmap(weight_val_df, annot=True, fmt=".3f",cmap="YlGnBu")

def IoU_classes(n_classes,weight_values, classes):
    """ Calculate IoU for each class or label"""
    # initializing a dict to store all the IoU values for each label or class
    IoU_individual_classes = {}
    for i , j, label in zip(np.arange(n_classes), np.arange(n_classes), classes):
        #IoU_individual_classes["classes_{0}".format(i)] = weight_values[i,j]/(np.sum(weight_values[:,i]) + np.sum(weight_values[j]) - weight_values[i,j])
        IoU_individual_classes[label] = weight_values[i,j]/(np.sum(weight_values[:,i]) + np.sum(weight_values[j]) - weight_values[i,j])
    IoU_all_classes = pd.DataFrame([IoU_individual_classes])
    return IoU_all_classes

IoU_classes_3 = IoU_classes(n_classes,weight_values_3,classes)
filename_csv = 'IoU_classes_UNET_resnet_152_300ep.csv'
IoU_classes_3.to_csv(filename_csv)
print(IoU_classes_3)