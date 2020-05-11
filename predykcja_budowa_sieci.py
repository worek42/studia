#Biblioteki do wczytania i formatowania danych
import csv
import numpy as np

#Bibioteki do obliczen tensorowych
import tensorflow as tf
from tensorflow import keras

#Bibioteka do obsługi sieci neuronowych
import keras

#-------------------------------Wczytywanie danych z bliku csv-----------------------------

def cases_array():
    temp = False
    cases = np.empty((50))
    i = 0 

    with open('owid-covid-data.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if(row['location'] == 'Poland'):

                if(row['date'] == '2020-03-15'):
                    temp=True

                if(row['date'] == '2020-05-04'):
                    break

                if(temp):
                    print(row['date'], row['total_cases'])
                    cases[i] = row['total_cases']
                    i += 1
    #print(cases)
    return cases

#----------------------------Przygotowanie baz dla sieci neuronowej----------------------

total_cases = cases_array()
print(max(total_cases))
total_cases = total_cases/20000


BazaAns = np.empty((43))
for i in range(7,50):
    BazaAns[i-7] = total_cases[i]

CasesCount = 43
CasesInput = 7
BazaCases = np.empty((CasesCount,CasesInput))

for i in range(7,50):
    BazaCases[i-7,:] = total_cases[i-7:i]


print(BazaCases)
#print(BazaAns)

#-------------------------------Stworzenia modelu sieci----------------------------------

input  = keras.engine.input_layer.Input(shape=(CasesInput,),name="wejscie")
#FlattenLayer = keras.layers.Flatten()
path = input

for i in range(0,10):
  LayerDense1 = keras.layers.Dense(20, activation=None, use_bias=True, kernel_initializer='glorot_uniform')
  path = LayerDense1(path)

  LayerPReLU1 = keras.layers.PReLU(alpha_initializer='zeros', shared_axes=None)
  path = LayerPReLU1(path)

LayerDenseN = keras.layers.Dense(1, activation=keras.activations.sigmoid, use_bias=True, kernel_initializer='glorot_uniform')
output = LayerDenseN(path)

#------------------------------Creation of TensorFlow Model-------------------------------

genderModel = keras.Model(input, output, name='genderEstimatior')
genderModel.summary() # Display summary

#-----------------------------Włączenia procesu uczenia-----------------------------------

rmsOptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
genderModel.compile(optimizer=rmsOptimizer,loss=keras.losses.mean_squared_error,metrics=['accuracy'])
genderModel.fit(BazaCases, BazaAns, epochs=30, batch_size=10, shuffle=True)
genderModel.save('siec_cases.h5')