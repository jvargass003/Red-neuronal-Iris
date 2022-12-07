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
import math  as m

class backpropagation:
    
    def __init__(self,xt,dt):
        #Generamos pesos sinápticos aleatorios
        w1 = np.random.randn(3,4)
        w2 = np.random.randn(3,3)
        filas, colum = np.shape(xt)
        
        
        for m in range(filas):
            x = np.reshape(xt[m,:],(4,1))
            d = np.reshape(dt[m],(3,1))
            
            #Propagación hacia adelante
            a1,a2,z2 = self.PropagacionAdelante(x,w1,w2)
            #Propagación hacia atras 
            s1, s2 = self.PropagacionAtras(a1,a2,d,z2,w2)
        
            w2 = self.actualizarPesos(w2,0.1,s2,a1)
            w1 = self.actualizarPesos(w1,0.1,s1,x)

        x = np.reshape(xt[9,:],(4,1))
        d = np.reshape(dt[9],(3,1))
        a1,a2,z2 = self.PropagacionAdelante(x,w1,w2)
        if a2[0]>0:
            print("Iris-setosa")
        else: 
            if a2[1]>0:
                print("Iris-versicolor")
            else:
                 print("Iris-virginica")
        print(d)
        
    def PropagacionAdelante(self,x,w1,w2):
        """
        Función. Realizamos la propagación hacia adelante, me base en la 
        ecuación f(w2*f(w1*x))
        Atributos: 
            x: Matriz  Valores de entrada de la red neuronal
        Return:
            a1: Matriz  Valores de activación de la capa 1
            a2: Matriz  Valores de activación de la capa 2
            z2: Martriz Valores de preactivación de la capa 2
            
        """
        #Agregar el valor de bias a las matrices x, w1, w2
        bx = self.agregarBias(x)
        bw1 = self.agregarBias(w1)
        bw2 = self.agregarBias(w2)
        
        #Calcular preactivación y activación con respecto a los pesos de la pri
        #primera capa
        z1 = self.Preactivacion(bw1,bx)
        a1 = self.funcionSigmoide(z1)
        
        #agregar el valor de bias a la matriz de activación 1
        ba1 = self.agregarBias(a1)
        
        #Valores de preactivación y activación de la capa 2
        z2 = self.Preactivacion(bw2,ba1)
        a2 = self.funcionReLu(z2)
        
        return a1,a2,z2
    
    def PropagacionAtras(self,a1,a2,d,z2,w2):
        """
        Función. Se calculan los valores del error y se propaga hacia atras
        
        Atributos: 
            a1: Matriz  Valores de activación de la capa 1
            a2: Matriz  Valores de activación de la capa 2
            z2: Martriz Valores de preactivación de la capa 2
            d: Matriz   Valores reales de la activación 
            w2: Matriz  Valores de pesos sináapticas
        
        Return:
            s1: Matriz  Valor de Delta, valor de actualización capa 1
            s2: Matriz  Valor de Delta, valor de actualización capa 2
            
        """
        #Calcular error de la salida
        e2 =self.calcularErrorCapaF(d,a2)
        #Calcular la derivada de la función de salida ReLu
        devReLu = self.derivadaReLu(z2)
        #Calcular el valor de actualización de valores capa1
        s2 = self.Delta(devReLu, e2)
        
        #Calcular el error de la activación capa 1
        e1=self.errorCapas(w2, s2)
        #Calcular derivada de la función de las capas ocultas
        devSig = self.devSigmoide(a1)
        #Calcular el valor de actualización de valores capa 1
        s1 = self.Delta(devSig,e1)
        
        return s1, s2
    
    def Preactivacion(self,w,x):
        """
        Función: Permite calcular los valores de preactivación haciendo el producto 
        de W * x
        
        Prametros: 
            W: Matriz  Valores de pesos aleatorios 
            X: Matriz  Valores de valores de activación
        Return 
            z: Matriz  Valores de preactivación 
        """
        z = np.matmul(w,x)
        return z

    def funcionSigmoide(self, z):
        """
        Función. Permite calcular el valor de activación de la preactivación 
        
        Parametros: 
            z: Matriz  Valores de Preactivación 
        
        Return:
            a: Matriz  Valores de activación
        
        """
        a = []
        f, c = np.shape(z)
        for x in range (f):
            a.append(1/(1+m.exp(-z[x,:])))
        a = np.reshape(a,(f,c))
        return a
        
    def agregarBias(self, x):
        """
        Función. Permite agregar el valor de bias a los valores de activación 
        
        Parametros:
            x: Matriz  Valores de activación 
        
        Return 
            bx: Matriz  Valores con el valor de bias agregado 
        """
        f, c = np.shape(x)
        if(c == 1):  
            bx = [[1]]
            bx = np.vstack([bx,x])
        else:
            bx = []
            for r in range(f):
                bx.append(1)
            bx = np.reshape(bx,(f,1))
            bx = np.append(bx,x, axis=1)
        return bx
    
    def funcionReLu(self, z):
        """
        Función. Permite calcular los valores de activación para la ultima capa
        
        Atributo:
            z: Matriz  Valores de preactivación 
        
        Return:
            a: Matriz  Valores resultantes de la función ReLu
            
        
        """
        a = []
        f, c = np.shape(z)
        for x in range (f):
            if(z[x,:]>=0):
                a.append(z[x,:])
            else:
                a.append(0)
        a = np.reshape(a,(f,c))
        return a
    
    def calcularErrorCapaF(self, d, a):
        """
        Función. Calcula el error de la ultima capa e = d-a
        
        Atributo:
            d: Matriz  Valores reales de los valores 
            a: Matriz  Valores resultantes después de la propagación hacia adelante
        Return:
            e: Matriz Valores de errores según el valor real y el calculado 
        """
        e = []
        f, c = np.shape(d)
        for x in range (f):
            e.append(d[x]-a[x,:])
        e = np.reshape(e,(f,c))
        return e
    
    def derivadaReLu(self,z):
        """
        Función. Calcula el valor de la derivada de ReLu, si el valor de z 
        es menor a 0 devuelve un 0 si no devuelve un 1
        
        Atributos:
            z: Matriz   Valores de preactivación de a
            
        Return:
            devReLu: Matriz  Valores calculados de la derivada de ReLu
            
        """
        devReLu = []
        f, c = np.shape(z)
        for x in range (f):
            if(z[x,:]>=0):
                devReLu.append(1)
            else:
                devReLu.append(0)
        devReLu = np.reshape(devReLu,(f,c))
        return devReLu
    
    def Delta(self, dev, e):
        """
        Función. Calcula el valor de Delta multiplicando los valor de la 
        derivada de la función de activación y los valores del error
        
        Atributos: 
            dev: Matriz  Valores de la derivada de la función de activación
            e: Matriz    Valores del error
        
        Return:
            s: Matriz  Valores de delta calculados 
        """
        
        delta = dev * e;
        return delta
    
    def errorCapas(self, w,s):
        """
        Función. Calcula el error para las todas las capas-1
        
        Atributo: 
            w: Matriz  Valores de los pesos sinápticos
            s: Matriz  Valores de Delta (f´(z)*e)
            
        Return:
            e: Matriz  Valores de errores calculado wT*s
        """
        
        wT = np.transpose(w)
        e = np.matmul(wT,s)
        return e
    
    def devSigmoide(self,a):
        """
        Función. Calcula la derivada de la función Sigmoide, dev = a * (1-a)
        
        Atributos: 
            a: Matriz  Valores de activción de las capas
            
        Return:
            dev: Matriz  Valores de la derivada de la función Sigmoide
        """
        
        dev = []
        f, c = np.shape(a)
        for x in range (f):
            dev.append(a[x]*(1-a[x]))
        dev = np.reshape(dev,(f,c))
        return dev
    
    def actualizarPesos(self, w,n,s,a):
        """
        Función. Actuliza los pesos, w-nsaT
        
        Atributos:
            w: Matriz  Valores de los pesos sinápticos propuestos
            n: Float   Valor de apendizaje
            s: Matriz  Valores de Delta
            a: Matriz  Valores de activación 
         Return:
            w: Matriz  Nuevos pesos sinápticos actualizados
        """
        wa = []
        aT = np.transpose(a)
        act = n * np.matmul(s,aT)
        wa = w + act
        return wa
        
        
        