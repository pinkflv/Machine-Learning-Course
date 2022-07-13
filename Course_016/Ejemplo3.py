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


df = pd.read_csv("http://virtualfif.uaq.mx/diplomado/data/practica_2/kc_house_data.csv")

df.head()
df.info()
df.describe()

plt.suptitle("Squareft per price")
plt.ticklabel_format(style='plain')
plt.scatter(df["sqft_living15"], df["price"])
plt.xlabel("price")
plt.ylabel("sqft_living15")
plt.show()

X = df.iloc[:, 2:3]
y = df.iloc[:, [-2]]

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

svr = SVR(C=1)

svr.fit(X, y)
svr.score(X, y)

y_pred = svr.predict(X) 

plt.suptitle("Squareft per price")
plt.ticklabel_format(style='plain')
plt.scatter(y, X)
plt.scatter(y_pred, X, color="red")
plt.xlabel("price")
plt.ylabel("sqft_living15")
plt.show()
