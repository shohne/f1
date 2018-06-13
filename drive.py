import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, LEFT, RIGHT, SPACE
from getkeys import key_check
from collections import deque, Counter
import numpy as np
import random

import keras
from keras.layers import Activation, Input, Dense, Embedding, LSTM, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Flatten, Concatenate, Dropout
from keras.models import Model
from keras.models import load_model
from keras.layers.convolutional import Cropping2D
from keras.layers.core import Lambda
from keras import optimizers
from keras.models import load_model
import keras.backend as K
import h5py
from keras import __version__ as keras_version

import random
import os


image_shape = None

model_left  = load_model('model_left.h5')
model_right = load_model('model_right.h5')
model_space = load_model('model_space.h5')

v = 0

move = [0,0,0]
isLeftPressedKey = False
isRightPressedKey = False
isSpacePressedKey = False

window_upleft = None
window_bottomright = None

while(True):
    keys = key_check()
    if 'T' in keys or 't' in keys:
        break

    screen = grab_screen(region=(0,0,1300,850))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    if window_upleft is None:
        cv2.imwrite('s.png', screen)
        screen = cv2.imread('s.png',0)
        marca = cv2.imread("marca.png",0)
        result = cv2.matchTemplate(screen, marca, cv2.TM_CCOEFF_NORMED)
        origin = np.unravel_index(result.argmax(),result.shape)
        print('origin', origin)
        window_upleft = [origin[0] + 45, origin[1]]
        print('window_upleft', window_upleft)

        marca_fim = cv2.imread("marca_fim.png",0)
        result = cv2.matchTemplate(screen, marca_fim, cv2.TM_CCOEFF_NORMED)
        position_right = np.unravel_index(result.argmax(),result.shape)

        print('position_right', position_right)

        window_bottomright = [position_right[1]*0.6, position_right[1]+190]
        print('window_bottomright', window_bottomright)

        horizontal_size = window_bottomright[1] - window_upleft[1] + 0
        vertical_size = int(horizontal_size * 0.56)
        image_shape = (vertical_size,horizontal_size,1)

        print('image_shape', image_shape)

        for i in range(-10,10):
            for j in range(-10,10):
                screen[origin[0]+i][origin[1]+j] = 0
#                screen[position_right[1]+i][position_right[0]+j] = 0
#                screen[500+i][500+j] = 0
        cv2.imwrite('verify' +  os.sep + 'v.png', screen)

        exit()

    np_image = screen[window_upleft[0]:window_upleft[0]+image_shape[0], window_upleft[1]:window_upleft[1]+image_shape[1]]
    np_image = np.reshape(np_image, (1, image_shape[0], image_shape[1], image_shape[2]) )

    prediction_left  = model_left.predict(np_image)
    prediction_right = model_right.predict(np_image)
    prediction_space = model_space.predict(np_image)


    print ('prediction_left',  prediction_left[0])
    print ('prediction_right', prediction_right[0])
    print ('prediction_space', prediction_space[0])

    if prediction_left[0][0] > random.random():
        if not isLeftPressedKey:
            print ('LEFT')
            PressKey(LEFT)
            isLeftPressedKey = True
            move[0] = 1
    else:
        if isLeftPressedKey:
            print ('RLEFT')
            ReleaseKey(LEFT)
            isLeftPressedKey = False
            move[0] = 0



    if prediction_right[0][0] > random.random():
        if not isRightPressedKey:
            print ('RIGHT')
            PressKey(RIGHT)
            isRightPressedKey = True
            move[1] = 1
    else:
        if isRightPressedKey:
            print ('RRIGHT')
            ReleaseKey(RIGHT)
            isRightPressedKey = False
            move[1] = 0



    if prediction_space[0][0] > random.random():
        if not isSpacePressedKey:
            print ('SPACE')
            PressKey(SPACE)
            isSpacePressedKey = True
            move[2] = 1
    else:
        if isSpacePressedKey:
            print ('RSPACE')
            ReleaseKey(SPACE)
            isSpacePressedKey = False
            move[2] = 0

    time.sleep(.15)

    if random.random() > 0.93:
        i = np.reshape(np_image, (image_shape[0], image_shape[1], image_shape[2]) )
        cv2.imwrite('verify' +  os.sep + 'v' + str(v) + '_' + str(move[0]) + str(move[1]) + str(move[2])  +  '.png', i)
        v = v + 1
