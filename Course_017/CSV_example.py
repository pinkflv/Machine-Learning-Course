# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 20:21:14 2022

@author: pinkf
"""

# importar librerias 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# leer los archivos de prueba
df_train = pd.read_csv("http://virtualfif.uaq.mx/diplomado/data/practica_3/train.csv")
df_test = pd.read_csv("http://virtualfif.uaq.mx/diplomado/data/practica_3/test.csv")
df_example = pd.read_csv("http://virtualfif.uaq.mx/diplomado/data/practica_3/example.csv")

# Analisis del la data en train
df_train.head()
df_train.info()
df_test.info()

# Limpieza de los datos nulos e innecesarios
df_train = df_train.drop(['Cabin', 'Embarked'],axis=1)
df_train = df_train.dropna(subset=['Age'])
df_train.isna().sum()

df_train.info()

# Preprocesado para el train DF
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder() 
df_train.iloc[:, 4]= labelencoder.fit_transform(df_train.iloc[:, 4].values)

X = df_train.iloc[:, [4,5]].values
y = df_train.iloc[:, 1].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/4,random_state=666)


# Paso 3 - Creacion de modelo

from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state=0)
log.fit(X_train,y_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

from sklearn.tree import DecisionTreeClassifier
arbol  = DecisionTreeClassifier(criterion="entropy", random_state=666)
arbol.fit(X_train,y_train)

from sklearn.ensemble import RandomForestClassifier
bosque = RandomForestClassifier(n_estimators=20, criterion="entropy", random_state=666)
bosque.fit(X_train,y_train)

from sklearn.svm import SVC
svm = SVC(kernel="linear", random_state=666)
svm.fit(X_train,y_train)

kernel = SVC(kernel="rbf", degree=3, random_state=666)
kernel.fit(X_train,y_train)

log.score(X_train,y_train)
knn.score(X_train,y_train)
arbol.score(X_train,y_train)
bosque.score(X_train,y_train)

log.score(X_test,y_test)
knn.score(X_test,y_test)
arbol.score(X_test,y_test)
bosque.score(X_test,y_test)

# acuraccy, precision, recall, F1
y_pred_log = log.predict(X_train)
y_pred_knn = knn.predict(X_train)
y_pred_arb = arbol.predict(X_train)
y_pred_bos = bosque.predict(X_train)
y_pred_svm = svm.predict(X_train)
y_pred_ker = kernel.predict(X_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("log - accuracy - Train: ",accuracy_score(y_train, y_pred_log)) 
print("log - precision - Train: ",precision_score(y_train, y_pred_log)) 
print("log - recall - Train: ",recall_score(y_train, y_pred_log))
print("log - f1 - Train: ",f1_score(y_train, y_pred_log))

print("knn - accuracy - Train: ",accuracy_score(y_train, y_pred_knn)) 
print("knn - precision - Train: ",precision_score(y_train, y_pred_knn)) 
print("knn - recall - Train: ",recall_score(y_train, y_pred_knn))
print("knn - f1 - Train: ",f1_score(y_train, y_pred_knn))

print("arbol - accuracy - Train: ",accuracy_score(y_train, y_pred_arb)) 
print("arbol - precision - Train: ",precision_score(y_train, y_pred_arb)) 
print("arbol - recall - Train: ",recall_score(y_train, y_pred_arb))
print("arbol - f1 - Train: ",f1_score(y_train, y_pred_arb))

print("bosque - accuracy - Train: ",accuracy_score(y_train, y_pred_bos)) 
print("bosque - precision - Train: ",precision_score(y_train, y_pred_bos)) 
print("bosque - recall - Train: ",recall_score(y_train, y_pred_bos))
print("bosque - f1 - Train: ",f1_score(y_train, y_pred_bos))

print("svm - accuracy - Train: ",accuracy_score(y_train, y_pred_svm)) 
print("svm - precision - Train: ",precision_score(y_train, y_pred_svm)) 
print("svm - recall - Train: ",recall_score(y_train, y_pred_svm))
print("svm - f1 - Train: ",f1_score(y_train, y_pred_svm))

print("kernel - accuracy - Train: ",accuracy_score(y_train, y_pred_ker)) 
print("kernel - precision - Train: ",precision_score(y_train, y_pred_ker)) 
print("kernel - recall - Train: ",recall_score(y_train, y_pred_ker))
print("kernel - f1 - Train: ",f1_score(y_train, y_pred_ker))

#Paso 4 - Evaluacion

y_pred_log = log.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_arb = arbol.predict(X_test)
y_pred_bos = bosque.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_ker = kernel.predict(X_test)


print("------------------------------------------ log")
print("log - accuracy - Test: ",accuracy_score(y_test, y_pred_log)) 
print("log - precision - Test: ",precision_score(y_test, y_pred_log)) 
print("log - recall - Test: ",recall_score(y_test, y_pred_log))
print("log - f1 - Test: ",f1_score(y_test, y_pred_log))

print("------------------------------------------ knn")
print("knn - accuracy - Test: ",accuracy_score(y_test, y_pred_knn)) 
print("knn - precision - Test: ",precision_score(y_test, y_pred_knn)) 
print("knn - recall - Test: ",recall_score(y_test, y_pred_knn))
print("knn - f1 - Test: ",f1_score(y_test, y_pred_knn))

print("------------------------------------------ arbol")
print("arbol - accuracy - Test: ",accuracy_score(y_test, y_pred_arb)) 
print("arbol - precision - Test: ",precision_score(y_test, y_pred_arb)) 
print("arbol - recall - Test: ",recall_score(y_test, y_pred_arb))
print("arbol - f1 - Test: ",f1_score(y_test, y_pred_arb))

print("------------------------------------------ bosque")
print("bosque - accuracy - Test: ",accuracy_score(y_test, y_pred_bos)) 
print("bosque - precision - Test: ",precision_score(y_test, y_pred_bos)) 
print("bosque - recall - Test: ",recall_score(y_test, y_pred_bos))
print("bosque - f1 - Test: ",f1_score(y_test, y_pred_bos))

print("------------------------------------------ svm")
print("svm - accuracy - Train: ",accuracy_score(y_test, y_pred_svm)) 
print("svm - precision - Train: ",precision_score(y_test, y_pred_svm)) 
print("svm - recall - Train: ",recall_score(y_test, y_pred_svm))
print("svm - f1 - Train: ",f1_score(y_test, y_pred_svm))

print("------------------------------------------ kernel")
print("kernel - accuracy - Train: ",accuracy_score(y_test, y_pred_ker)) 
print("kernel - precision - Train: ",precision_score(y_test, y_pred_ker)) 
print("kernel - recall - Train: ",recall_score(y_test, y_pred_ker))
print("kernel - f1 - Train: ",f1_score(y_test, y_pred_ker))



# Preprocesado para el train DF
df_test = df_test.dropna(subset=['Age'])

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder() 
df_train.iloc[:, 4]= labelencoder.fit_transform(df_train.iloc[:, 4].values)

# Tomamos las columnas que tienen datos similares a los valores con los que 
# entrenamos0
A = df_test.iloc[:, [4,5]].values

print(A)

# Escalamos los valores para poder usar el modelo de prediccion elegido
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data_for_pred = sc.fit_transform(A)

# Hacemos la prediccion
pred = arbol.predict(data_for_pred)

print(pred)

# Exportar CSV
def download_output(pred, name):
  output = pd.DataFrame({'Edad': df_test.iloc[:, 4].values,
                         'Genero': df_test.iloc[:, 5].values,
                         'Survived': pred})
  output.to_csv(name)
  
# Se exporta el CSV
download_output(pred, 'CSV_example.csv')

