# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 20:21:14 2022

@author: pinkf
"""
# Importacion de librerias
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# DataFrames obtenidos de un link (pueden cambiarse por un csv descargado)
df = pd.read_csv("http://virtualfif.uaq.mx/diplomado/data/practica_3/train.csv")
df_test = pd.read_csv("http://virtualfif.uaq.mx/diplomado/data/practica_3/test.csv") 

# Seleccion de datos y division para su entrenamiento
X = df.iloc[:,[2,9]].values
y = df.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/4,random_state=0)

# Eleccion del modelo a usar, revisando con otros modelos el modelo de arbol me
# parecio mas adecuado
from sklearn.tree import DecisionTreeClassifier
arbol  = DecisionTreeClassifier(criterion="entropy", random_state=666)
arbol.fit(X_train,y_train)

# Seleccion y preparacion de datos para el testeo
X_test = df_test.iloc[:,[1,8]].values
X_test = np.nan_to_num(X_test)
arbol.score(X_train,y_train)

# Se hace la prediccion usando el modelo de arbol
y_pred_arbol = arbol.predict(X_test)

print(y_pred_arbol.size)

# Exportar CSV
def download_output(pred, name):
  output = pd.DataFrame({'PassengerId': df_test.iloc[:, 0].values,
                         'Survived': pred})
  output.to_csv(name, index=False)
  
# Se exporta el CSV
download_output(y_pred_arbol, 'CSV_example.csv')