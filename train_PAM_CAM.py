# Author: Cunjian Chen (ccunjian@gmail.com), Updated at 2/25/2021
# PAM+CAM:https://github.com/niecongchong/DANet-keras/blob/master/layers/attention.py
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from attention import PAM, CAM
# use single gpu will result in poor performance
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.layers import Flatten, Dense, AveragePooling2D,GlobalAveragePooling2D,Activation,Dropout, Reshape, Multiply, Add, Conv2D, BatchNormalization, Concatenate
from keras.models import Model
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from time import time
import utils

from keras import backend as K
import tensorflow as tf
from keras.utils import multi_gpu_model

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', use_activation=True):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x

# Compatible with tensorflow backend
def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

learning_rate = 0.0001
img_width = 224
img_height = 224
#nbr_train_samples = 3019
#nbr_validation_samples = 758

nbr_epochs = 50
batch_size = 32


train_data_dir='' # train path to two folders with both live and PA
val_data_dir = '' # val path to two folders with both live and PA

nbr_train_samples = utils.get_num_files(train_data_dir)
nbr_validation_samples = utils.get_num_files(val_data_dir)

FishNames = ['live', 'spoof'] # change the name accordingly

print('Loading InceptionV3 Weights ...')
#InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
#                    input_tensor=None, input_shape=(299, 299, 3))
InceptionV3_notop = DenseNet121(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(224, 224, 3))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

print('Adding Average Pooling Layer and Softmax Output Layer ...')
output = InceptionV3_notop.get_layer(index = -1).output  # Shape: (7, 7, 2048) # for Densetnet: 1024; Resnet/Inception: 2048
#print output.shape


pam = PAM()(output)
pam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)
pam = BatchNormalization(axis=3)(pam)
pam = Activation('relu')(pam)
pam = Dropout(0.5)(pam)
pam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)

cam = CAM()(output)
cam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
cam = BatchNormalization(axis=3)(cam)
cam = Activation('relu')(cam)
cam = Dropout(0.5)(cam)
cam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)

feature_sum = Add()([pam, cam])
feature_sum = Dropout(0.5)(feature_sum)
#feature_sum = Dropout(0.5)(cam)
feature_sum = Conv2d_BN(feature_sum, 512, 1)
feature_sum = GlobalAveragePooling2D()(feature_sum)
attention_output = Dense(2, activation='softmax',kernel_initializer='he_normal')(feature_sum) # Resnet50 works better with he_normal here

InceptionV3_model = Model(InceptionV3_notop.input, attention_output)
InceptionV3_model.summary()

optimizer = Adam(lr = learning_rate)
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

# autosave best Model
best_model_file = "./weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 2, save_best_only = True)
#best_model = ModelCheckpoint(filepath = "saved_models/spoof_{epoch:02d}_{val_acc:.4f}h.h5", monitor='val_acc', verbose = 2, save_best_only = False)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = FishNames,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = FishNames,
        class_mode = 'categorical')

tensorboard=TensorBoard(log_dir="logs/{}".format(time()))

InceptionV3_model.fit_generator(
        train_generator,
        samples_per_epoch = nbr_train_samples,
        nb_epoch = nbr_epochs,
        validation_data = validation_generator,
        nb_val_samples = nbr_validation_samples,
        callbacks = [best_model,tensorboard])
'''
InceptionV3_model.save_weights('model_InceptionV3_SE.h5')
model_json = InceptionV3_model.to_json()
with open('model_InceptionV3_SE.json', "w") as json_file:
    json_file.write(model_json)
json_file.close()
'''
