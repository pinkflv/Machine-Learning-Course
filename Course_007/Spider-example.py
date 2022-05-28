# -*- coding: utf-8 -*-
"""
Created on Tue May 24 20:04:38 2022

@author: pinkf
"""

import pandas as pd
import matplotlib.pyplot as plt
dataFrame = pd.read_csv("toy_dataset.csv")

plt.suptitle("Grafica comparativa")
plt.title("Edad vs Ingresos")
plt.plot(dataFrame["Age"])
plt.grid(0)
plt.show()