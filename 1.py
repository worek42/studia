import tensorflow as tf
from tensorflow import keras
from glob import glob
import os
import imageio as im

import numpy as np

import keras

# załadowanie bazy uczącej


imgCount = 30
imgWidth = 50
imgHeight = 50

bazaImg = np.empty((imgCount, imgHeight, imgWidth, 3))

m_files = glob('./baza/m/.jpg')
k_files = glob('./baza/k/.jpg')

bazaAns = []

i=0
for file in m_files:
    img = im.imread(file)
    bazaImg[i,:,:,:] = img[0:imgHeight,0:imgWidth,0:3]
    i+=1
    bazaAns.append(0)
    if i>=imgCount/2:
        break

for file in k_files:
    img = im.imread(file)
    bazaImg[i,:,:,:] = img[0:imgHeight,0:imgWidth,0:3]
    i+=1
    bazaAns.append(1)
    if i>imgCount:
        break

# stworzenie modelu sieci

input = keras.engine.input_layer.Input(shape=(imgHeight,imgWidth,3), name="wejscie")

FlattenLayer = keras.layers.Flatten()

for i in range (0,3):
    path = FlattenLayer(input)
    LayerDense1 = keras.layers.Dense(200,activation=None, use_bias=True, kernel_initializer='glorot_uniform')
    path = LayerDense1(path)

    LayerPreLU1 = keras.layers.PReLU(alpha_initializer='zeros', shared_axes=None)
    path = LayerPreLU1(path)

LayerDenseN = keras.layers.Dense(1, activation=keras.activations.sigmoid, use_bias=True, kernel_initializer='glorot_uniform')
output=LayerDenseN(path)

#tworzenie tensor flow model

genderModel = keras.Model(input, output, name='genderEstimator')

genderModel.summary()

# włączenie procesu uczenia
rmsOptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

genderModel.compile(optimizer=rmsOptimizer,loss=keras.losses.binary_crossentropy,metrics=['accuracy'])

genderModel.fit(bazaImg, bazaAns, epochs=100, batch_size=10, shuffle=True)

# przetestować