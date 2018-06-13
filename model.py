
import pandas as pd
import numpy as np
import keras
from keras.layers import Activation, Input, Dense, Embedding, LSTM, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Flatten, Concatenate, Dropout
from keras.models import Model
from keras.models import load_model
from keras.layers.convolutional import Cropping2D
from keras.layers.core import Lambda
from keras import optimizers
import keras.backend as K
from PIL import Image
import os.path
import os
import random
import socket
import json
import matplotlib.pyplot as plt
import cv2


DATA_DIRECTORY = 'data'
MINIBATCH_SIZE = 40
# split dataset in train (70%), validation (10%) and test (20%)
TRAIN_DATASET_SIZE = 0.8
VALIDATION_DATASET_SIZE = 0.18
NUM_EPOCHS = 100
DATASET_MAX_LENGTH = 1000000000
#DATASET_MAX_LENGTH = 256
LEARNING_RATE = 0.0185

df = None
files_in_data_directory = os.listdir(DATA_DIRECTORY)
for file in files_in_data_directory:
#    print ('file', file)
    if file.startswith('label_'):
        print(file)
        df_temp = pd.read_csv(DATA_DIRECTORY + os.sep + file, sep = ';', header = 1,
                                names = ['image_filename','left', 'right', 'space'])
        if df is None:
            df = df_temp
        else:
            df.append(df_temp)



# number of datapoints in whole dataset
dataset_length = df.shape[0]
print('dataset_length', dataset_length)

if dataset_length > DATASET_MAX_LENGTH:
    dataset_length = DATASET_MAX_LENGTH

# auxiliary index to split and shuffle dataset
idx = list(range(dataset_length))
random.seed(a=543375)
random.shuffle(idx)
number_of_datapoints_in_train_dataset = int(dataset_length * TRAIN_DATASET_SIZE)
number_of_datapoints_in_validation_dataset = int(dataset_length * VALIDATION_DATASET_SIZE)
number_of_datapoints_in_test_dataset = dataset_length - number_of_datapoints_in_train_dataset - number_of_datapoints_in_validation_dataset

# create auxiliary index for each dataset
idx_train = idx[:number_of_datapoints_in_train_dataset]
idx_validation = idx[number_of_datapoints_in_train_dataset:(number_of_datapoints_in_train_dataset+number_of_datapoints_in_validation_dataset)]
idx_test = idx[(number_of_datapoints_in_train_dataset+number_of_datapoints_in_validation_dataset):]

# figure out image size
file_image = DATA_DIRECTORY +  os.sep + os.sep + df.iloc[0]['image_filename']
image = cv2.imread(file_image,0)
print ('image.shape1', image.shape)

marca = cv2.imread("marca.png",0)
result = cv2.matchTemplate(image, marca, cv2.TM_CCOEFF_NORMED)
origin = np.unravel_index(result.argmax(),result.shape)
window_upleft = [origin[0] + 45, origin[1]]
print('window_upleft', window_upleft)

marca_fim = cv2.imread("marca_fim.png",0)
result = cv2.matchTemplate(image, marca_fim, cv2.TM_CCOEFF_NORMED)
position_right = np.unravel_index(result.argmax(),result.shape)
window_bottomright = [position_right[1]*0.6, position_right[1]+190]
print('window_bottomright', window_bottomright)

horizontal_size = window_bottomright[1] - window_upleft[1] + 0
vertical_size = int(horizontal_size * 0.56)
image_shape = (vertical_size,horizontal_size,1)

print ('image_shape', image_shape)

# image = image[window_upleft[0]:window_upleft[0]+image_shape[0], window_upleft[1]:window_upleft[1]+image_shape[1]]


# generetor for train, validation and test datasets
def get_data_generador(df, idx, chunk_size, image_shape, control, type_dataset):
    idx_base_chunk = 0
    print ('control', control, 'type_dataset', type_dataset)


    while True:
        idx_dataset = idx_train
        if type_dataset == 'validation':
            idx_dataset = idx_validation
        if type_dataset == 'test':
            idx_dataset = idx_test

        this_chunk_size = chunk_size
        if idx_base_chunk + this_chunk_size > len(idx_dataset):
            this_chunk_size = len(idx_dataset) - idx_base_chunk
        np_image = np.zeros((this_chunk_size, image_shape[0], image_shape[1], 1))
        np_label = np.zeros((this_chunk_size, 1))
        for i in range(this_chunk_size):
            file_image = DATA_DIRECTORY + os.sep + df.iloc[idx_dataset[i+idx_base_chunk]]['image_filename']
#            print ('file_image', file_image)
            space = df.iloc[idx_dataset[i+idx_base_chunk]]['space']
            right = df.iloc[idx_dataset[i+idx_base_chunk]]['right']
            left = df.iloc[idx_dataset[i+idx_base_chunk]]['left']
            image = cv2.imread(file_image,0)
            image = image[window_upleft[0]:window_upleft[0]+image_shape[0], window_upleft[1]:window_upleft[1]+image_shape[1]]
            image = np.reshape(image, image_shape)
            np_image[i+0,:,:,:] = image
            np_label[i,0] = 0
            if control == 'left':
                if left == 1:
                    np_label[i,0] = 1
            if control == 'right':
                if right == 1:
                    np_label[i,0] = 1
            if control == 'space':
                if space == 1:
                    np_label[i,0] = 1
#            print ('label', np_label[i,0])

        idx_base_chunk = idx_base_chunk + this_chunk_size
        if (idx_base_chunk == len(idx_dataset)):
            idx_base_chunk = 0
#        print ('np_label', np_label)
        yield np_image, np_label


def buildModel(image_shape, name_model):
    input_image = Input(shape = image_shape, name = 'image')
    x = input_image
    x = Lambda(lambda x: (x/255.0)-0.5)(x)

    x = Conv2D(12, kernel_size = (10,10), strides  = (1,1), padding = 'same', name = 'conv', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = AveragePooling2D((3, 6), strides=(2, 4), name = 'average')(x)

    x = Conv2D(24, kernel_size = (10,10), strides  = (1,1), padding = 'same', name = 'conv_1', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = Activation('relu', name = 'relu_1')(x)
    x = MaxPooling2D((3, 6), strides=(2, 4), name = 'maxpool_1')(x)

    x = Conv2D(36, kernel_size = (7,7), strides  = (1,1), padding = 'same', name = 'conv_2', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = Activation('relu', name = 'relu_2')(x)
    x = MaxPooling2D((3, 6), strides=(2, 4), name = 'maxpool_2')(x)

    x = Conv2D(48, kernel_size = (5,5), strides  = (1,1), padding = 'same', name = 'conv_3', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = Activation('relu', name = 'relu_3')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name = 'maxpool_3')(x)

    x = Conv2D(64, kernel_size = (3,3), strides  = (1,1), padding = 'same', name = 'conv_4', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = Activation('relu', name = 'relu_4')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name = 'maxpool_4')(x)


    x = Flatten()(x)
    fc1 = Dense(512, name='fc1', kernel_initializer = keras.initializers.glorot_uniform())(x)
    fc1 = Activation('relu')(fc1)
#    fc1 = Dropout(0.5, name = 'dropout_fc1')(fc1)

    fc3 = Dense(16, name='fc3', kernel_initializer = keras.initializers.glorot_uniform())(fc1)

    predict_control = Dense(1, name='predict_control', kernel_initializer = keras.initializers.glorot_uniform())(fc3)
    predict_control = Activation('sigmoid')(predict_control)

    model = Model(inputs = input_image, outputs = predict_control, name=name_model)

    return model


# split indicex dataset
idx_train = idx[:number_of_datapoints_in_train_dataset]
idx_validation = idx[number_of_datapoints_in_train_dataset:(number_of_datapoints_in_train_dataset+number_of_datapoints_in_validation_dataset)]
idx_test = idx[(number_of_datapoints_in_train_dataset+number_of_datapoints_in_validation_dataset):]

sgd = optimizers.SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)


model_left = None
if os.path.isfile('model_left.h5'):
    model_left = load_model('model_left.h5')
else:
    model_left = buildModel(image_shape, 'model_left')
print ('model_left', model_left)
model_left.summary()
model_left.compile(optimizer=sgd,  loss='binary_crossentropy',  metrics=['binary_crossentropy', 'accuracy'])
callbacks_left_list = [
    keras.callbacks.ModelCheckpoint(filepath='model_left.h5', monitor='val_loss', save_best_only=False)
]
history_left = model_left.fit_generator(
    get_data_generador(df, idx_train, MINIBATCH_SIZE, image_shape, 'left',  'train'),
    steps_per_epoch = int(number_of_datapoints_in_train_dataset / MINIBATCH_SIZE) + 1,
    epochs = NUM_EPOCHS,
    validation_data = get_data_generador(df, idx_validation, MINIBATCH_SIZE, image_shape, 'left', 'validation'),
    validation_steps = int(number_of_datapoints_in_validation_dataset / MINIBATCH_SIZE) + 1,
    callbacks = callbacks_left_list
)
model_left.save('model_left.h5')


model_right = None
if os.path.isfile('model_right.h5'):
    model_right = load_model('model_right.h5')
else:
    model_right = buildModel(image_shape, 'model_right')
print ('model_right', model_right)
model_right.summary()
model_right.compile(optimizer=sgd,  loss='binary_crossentropy',  metrics=['binary_crossentropy', 'accuracy'])
callbacks_right_list = [
    keras.callbacks.ModelCheckpoint(filepath='model_right.h5', monitor='val_loss', save_best_only=False)
]
history_right = model_right.fit_generator(
    get_data_generador(df, idx_train, MINIBATCH_SIZE, image_shape, 'right', 'train'),
    steps_per_epoch = int(number_of_datapoints_in_train_dataset / MINIBATCH_SIZE) + 1,
    epochs = NUM_EPOCHS,
    validation_data = get_data_generador(df, idx_validation, MINIBATCH_SIZE, image_shape, 'right', 'validation'),
    validation_steps = int(number_of_datapoints_in_validation_dataset / MINIBATCH_SIZE) + 1,
    callbacks = callbacks_right_list
)
model_right.save('model_right.h5')



model_space = None
if os.path.isfile('model_space.h5'):
    model_space = load_model('model_space.h5')
else:
    model_space = buildModel(image_shape, 'model_space')
print ('model_space', model_space)
model_space.summary()
model_space.compile(optimizer=sgd,  loss='binary_crossentropy',  metrics=['binary_crossentropy', 'accuracy'])
callbacks_space_list = [
    keras.callbacks.ModelCheckpoint(filepath='model_space.h5', monitor='val_loss', save_best_only=False)
]
history_space = model_space.fit_generator(
    get_data_generador(df, idx_train, MINIBATCH_SIZE, image_shape, 'space', 'test'),
    steps_per_epoch = int(number_of_datapoints_in_train_dataset / MINIBATCH_SIZE) + 1,
    epochs = NUM_EPOCHS,
    validation_data = get_data_generador(df, idx_validation, MINIBATCH_SIZE, image_shape, 'space', 'validation'),
    validation_steps = int(number_of_datapoints_in_validation_dataset / MINIBATCH_SIZE) + 1,
    callbacks = callbacks_space_list
)
model_space.save('model_space.h5')




fig = plt.figure()
plt.plot(history_left.history['loss'])
plt.plot(history_left.history['val_loss'])
plt.plot(history_right.history['loss'])
plt.plot(history_right.history['val_loss'])
plt.plot(history_space.history['loss'])
plt.plot(history_space.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
fig.savefig('train_history.png')
