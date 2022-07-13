# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:54:46 2022

@author: pinkf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("http://virtualfif.uaq.mx/diplomado/data/regresion/Poli/Position_Salaries.csv")

# 1.- PDA

df.head()
df.info()
df.describe()

plt.suptitle("Salarios vs Posicion")
plt.title("Datos", color="red")
plt.scatter(df["Level"], df["Salary"])
plt.xlabel("Nivel")
plt.ylabel("Salario")
plt.grid(1)
plt.show()

### Outliners
### Escalados


X = df.iloc[:, 1:2]
y = df.iloc[:, [-1]]


sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Aqui dividiremos los datos de entrenamiento/test

# Creacion del modelo
svr = SVR(C=1)

svr.fit(X, y)
svr.score(X, y)

 # Uso del modelo
y_pred = svr.predict(X) 
 
plt.suptitle("Salarios vs Posicion")
plt.title("Datos", color="red")
plt.plot(X, y_pred)
plt.scatter(X, y)
plt.xlabel("Nivel")
plt.ylabel("Salario")
plt.grid(1)
plt.show()