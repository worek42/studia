#Biblioteki do wczytania i formatowania danych
import csv
import numpy as np

#Bibioteki do obliczen tensorowych
import tensorflow as tf
from tensorflow import keras

#Bibioteka do obsługi sieci neuronowych
import keras
from keras.models import load_model

#-------------------------------Wczytywanie danych z bliku csv-----------------------------

def cases_array():
    temp = False
    cases = np.empty((50))
    i = 0 

    with open('owid-covid-data.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if(row['location'] == 'Poland'):

                if(row['date'] == '2020-05-04'):
                    temp=True

                if(temp):
                    print(row['date'], row['total_cases'])
                    cases[i] = row['total_cases']
                    i += 1
                
                if(row['date'] == '2020-05-11'):
                    break
    print(i)
    return cases

#--------------------------------Załadowanie modelu---------------------------------------

genderModel = load_model('siec_cases.h5')
genderModel.summary() # Display summary

#---------------Przygotowanie wektora wejściowego dla sieci neuronowej--------------------

total_cases = cases_array()/20000

CasesInput = 7
BazaCases = np.empty((1,CasesInput))
BazaCases[0,:] = total_cases[0:7]

#----------------------------------Testowanie sieci---------------------------------------

gender = genderModel.predict(BazaCases)
print("Przewidywana wartość:"+str(int(gender[0]*20000))+"\nPoprawna wartość: "+str(total_cases[7]*20000))
text_file = open("predykcja_testowanie_sieci.txt", "w")
text_file.write("Przewidywana wartość: "+str(int(gender[0]*20000))+"\nPoprawna wartość: "+str(total_cases[7]*20000))
text_file.close()