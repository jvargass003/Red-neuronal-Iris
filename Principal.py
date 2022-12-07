"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO 
UA: REDES NEURONALES
TEMA:Backpropagation
Alumno: Jessica Vargas Sánchez 
Profesor: Dr. Asdrúbal López Chau
Descripción: Implental el algoritmo backpropagation para la función XOR

Created on Tue Nov 23 11:33:40 2021

@author: Jessica Vargas Sánchez
"""
import numpy as np
import pandas as pd
from Backpropagation import backpropagation 

datos = pd.read_csv("iris.csv")

x = np.array(datos.iloc[:,0:4])
y = np.array(datos.iloc[:,4:5])

f,c = np.shape(y)
d = []
for m in range(f):
    if (y[m] == "Iris-setosa"): 
        d.append([1,0,0])
    else: 
        if(y[m] == "Iris-versicolor"): 
            d.append([0,1,0])
        else:    
            d.append([0,0,1])

b = backpropagation(x,d)
