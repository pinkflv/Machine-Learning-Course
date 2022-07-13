# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:54:46 2022

@author: pinkf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("http://virtualfif.uaq.mx/diplomado/data/practica_2/EconomiesOfScale.csv")

# 1.- PDA

df.head()
df.info()
df.describe()

plt.suptitle("Economies of Scale")
plt.title("Units per cost", color="red")
plt.scatter(df["Number of Units"], df["Manufacturing Cost"])
plt.xlabel("Number of Units")
plt.ylabel("Manufacturing Cost")
plt.grid(1)
plt.show()

X = df.iloc[:,:-1].values
y = df.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
# Aqui dividiremos los datos de entrenamiento/test

# Creacion del modelo
poly = PolynomialFeatures(degree=6
                          )
X_poly = poly.fit_transform(X)

modelo = LinearRegression()

 # Uso del modelo
modelo.fit(X_poly, y)

modelo.score(X_poly, y)
y_pred = modelo.predict(X_poly)

plt.suptitle("Economies of Scale")
plt.title("Units per cost", color="red")
plt.scatter(X, y)
plt.scatter(X, y_pred, color="red")
plt.xlabel("Number of Units")
plt.ylabel("Manufacturing Cost")
plt.grid(1)
plt.show()
