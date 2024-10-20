#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src="https://i.ibb.co/v3CvVz9/udd-short.png" width="150"/>
#     <br>
#     <strong>Universidad del Desarrollo</strong><br>
#     <em>Magíster en Data Science</em><br>
#     <em>Profesor: Tomás Fontecilla </em><br>
# 
# </div>
# 
# # Machine Learning
# *12 de Octubre de 2024*
# 
# **Nombre Estudiante(s)**: `Giuseppe Lavarello - Víctor Saldivia - Ingrid Solís - Cristian Tobar`  

#Es probable que no les corra a la primera, yo tube que instalar/actualizar estos paquetes

#%pip install ucimlrepo
#%pip install xgboost
#%pip install --upgrade scikit-learn
#%pip install --upgrade imbalanced-learn


import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo 
import missingno as msng

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight

from imblearn.over_sampling import ADASYN

from xgboost import XGBClassifier

sns.set_theme()
pd.set_option('display.max_columns', None)


# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 
  
# metadata 
print(cdc_diabetes_health_indicators.metadata) 
  
# variable information 
print(cdc_diabetes_health_indicators.variables) 



df0=pd.concat([y,X], axis=1)


# Mostrar las primeras 10 filas de los datos

df0.head(10)


# Recopilación de información básica sobre el conjunto de datos
df0.info()


# Notar que estan todas las Features consideradas como Int. Se cambia a Bool donde corresponda

binary_features = {
    "Diabetes_binary": bool,
    "HighBP": bool,
    "HighChol": bool,
    "CholCheck": bool,
    "Smoker": bool,
    "Stroke": bool,
    "HeartDiseaseorAttack": bool,
    "PhysActivity": bool,
    "Fruits": bool,
    "Veggies": bool,
    "HvyAlcoholConsump": bool,
    "AnyHealthcare": bool,
    "NoDocbcCost": bool,
    "DiffWalk": bool,
    "Sex": bool,
}
df0 = df0.astype(binary_features)


# Recopilación de estadísticas descriptivas sobre los datos
df0.describe()


# Mostrar el tamaño del DataFrame
df0.shape


# Chequeo visual de Nans
msng.matrix(df0)


# Chequeo por dupes
duplicados = df0.duplicated().sum()

# Porcentage de dupes
porcentage = df0.duplicated().sum() / X.shape[0] * 100

print(f'{duplicados} filas contienen duplicados, lo que representa el {porcentage.round(2)}% del total de los datos.')


# Eliminar duplicados y guardar el dataframe resultante en una nueva variable
df1 = df0.drop_duplicates(keep='first')

df1.shape


# Crear un boxplot para visualizar la distribución de todas las variables numéricas y detectar posibles valores atípicos
fig, axs= plt.subplots(1, 2, figsize=(10,5))
# seleccionar columnas numéricas
num_columns = df1.select_dtypes(include=['number'])

# normalizar los valores usando escalado Min-Max
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(num_columns)

# Crear un dataframe con los datos normalizados
df_normalized = pd.DataFrame(normalized_data, columns=num_columns.columns)
sns.boxplot(data= df_normalized, ax=axs[0])
sns.violinplot(data= df_normalized, ax=axs[1])

axs[0].tick_params(axis='x', rotation=45)
axs[1].tick_params(axis='x', rotation=45)

fig.suptitle('Grafica de Outliers')
plt.show()


# Bucle para recorrer cada columna numérica y detectar y contar valores atípicos
out_cols=['BMI', 'GenHlth', 'MentHlth', 'PhysHlth']
upperlim = {}
for column in out_cols:
    print(f"Procesando columna: {column}")
    
    # Calcular el valor del percentil 25
    percentil25 = df1[column].quantile(0.25)
    
    # Calcular el valor del percentil 75
    percentil75 = df1[column].quantile(0.75)
    
    # Calcular el rango intercuartil (IQR)
    iqr = percentil75 - percentil25
    
    # Definir los límites superior e inferior para los valores no atípicos
    límite_superior = percentil75 + 1.5 * iqr
    límite_inferior = percentil25 - 1.5 * iqr
    upperlim[column] = límite_superior
    
    print(f"Límite inferior para {column}:", límite_inferior)
    print(f"Límite superior para {column}:", límite_superior)
    
    # Identificar los valores atípicos en la columna actual
    valores_atípicos = df1[(df1[column] > límite_superior) | (df1[column] < límite_inferior)]
    
    # Contar el número de filas con valores atípicos
    print(f"Número de filas que contienen valores atípicos en {column}: {len(valores_atípicos)}\n")


# **Desición** Basandonos en las figuras y las cantidades se descartara los outliers en BMI, MenHlth y PhysHlth, manteniendo los valores de GenHlth, pues se comportan de manera Normal

# usar una máscara booleana para eliminar valores atípicos
mask = True
for column, upper_limit in upperlim.items():
    if column == 'GenHelth': continue
    mask &= (df1[column] <= upper_limit)
df2 = df1[mask].copy()


# Crear un boxplot para visualizar la distribución de todas las variables numéricas y detectar posibles valores atípicos
fig, axs= plt.subplots(1, 2, figsize=(10,5))
# seleccionar columnas numéricas
num_columns = df2.select_dtypes(include=['number'])

# normalizar los valores usando escalado Min-Max
scaler = MinMaxScaler()
normalized_data2 = scaler.fit_transform(num_columns)

# Crear un dataframe con los datos normalizados
df_normalized2 = pd.DataFrame(normalized_data2, columns=num_columns.columns)
sns.boxplot(data= df_normalized2, ax=axs[0])
sns.violinplot(data= df_normalized2, ax=axs[1])

axs[0].tick_params(axis='x', rotation=45)
axs[1].tick_params(axis='x', rotation=45)

fig.suptitle('Grafica post trata de outliers')
plt.show()


# ## separacion de los datos

# Preparar características (X) y la variable objetivo (y)
X = df2.drop('Diabetes_binary', axis=1) 
y = df2['Diabetes_binary']


# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5555)

#Como existe un gran desbalance entre la incidencia de las clases tenemos que rebalancear
# ada = ADASYN(random_state=5555)
# X_train, y_train = ada.fit_resample(X_train, y_train)


# ## Regresión Logistica

def plt_conf_matrix(y_test,y_pred):
    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Graficar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.show()


# Crear y ajustar el modelo de Regresión Logística
logreg_model = LogisticRegression(random_state=5555, max_iter=1000, class_weight='balanced')  # Aumentar max_iter si es necesario
logreg_model.fit(X_train, y_train)

# Hacer predicciones
y_pred_logreg = logreg_model.predict(X_test)

# Evaluar el modelo
print(classification_report(y_test, y_pred_logreg))

plt_conf_matrix(y_test,y_pred_logreg)




# Importancias de las características
importances_reglog = np.array(np.abs(logreg_model.coef_[0])*X.std()) #Valor del coeficiente de cada variable, ponderado por la variación standard para mostrar el efecto "real"
indices = np.argsort(importances_reglog)[::-1]

# Graficar las importancias de las características
plt.figure()
plt.title('Importancias de las Características')
plt.bar(range(X.shape[1]), importances_reglog[indices], align='center')
plt.xticks(range(X.shape[1]), np.array(X.columns)[indices],rotation=90,)
plt.xlim([-1, X.shape[1]])
plt.show()


# Obtener las probabilidades predichas para la clase positiva
y_probs_logreg = logreg_model.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs_logreg)

# Calcular el AUC
roc_auc = roc_auc_score(y_test, y_probs_logreg)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Línea diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva Característica del Receptor (ROC)')
plt.legend(loc='lower right')
plt.grid()

plt.show()


# ## Arboles de Decision

# Crear y ajustar el árbol de desición
arbol = DecisionTreeClassifier(random_state=5555, max_depth= 10, class_weight='balanced')
arbol.fit(X_train,y_train)

# Hacer predicciones
y_preds_arbol = arbol.predict(X_test)

# Evaluar el modelo
print(classification_report(y_test, y_preds_arbol))

plt_conf_matrix(y_test,y_preds_arbol)




# Importancias de las características
importances_dtree = arbol.feature_importances_
indices = np.argsort(importances_dtree)[::-1]

# Graficar las importancias de las características
plt.figure()
plt.title('Importancias de las Características')
plt.bar(range(X.shape[1]), importances_dtree[indices], align='center')
plt.xticks(range(X.shape[1]), np.array(X.columns)[indices],rotation=90,)
plt.xlim([-1, X.shape[1]])
plt.show()


# Obtener las probabilidades predichas para la clase positiva
y_probs_arbol = arbol.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs_arbol)

# Calcular el AUC
roc_auc = roc_auc_score(y_test, y_probs_arbol)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Línea diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva Característica del Receptor (ROC)')
plt.legend(loc='lower right')
plt.grid()

plt.show()


# ## RndForest
# 

# Crear y ajustar el modelo de Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10 ,random_state=5555, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Hacer predicciones
y_pred_rndforest = rf_model.predict(X_test)

# Evaluar el modelo
print(classification_report(y_test, y_pred_rndforest))

plt_conf_matrix(y_test,y_pred_rndforest)




# Importancias de las características
importances_rf = rf_model.feature_importances_
indices = np.argsort(importances_rf)[::-1]

# Graficar las importancias de las características
plt.figure()
plt.title('Importancias de las Características')
plt.bar(range(X.shape[1]), importances_rf[indices], align='center')
plt.xticks(range(X.shape[1]), np.array(X.columns)[indices],rotation=90,)
plt.xlim([-1, X.shape[1]])
plt.show()


# Obtener las probabilidades predichas para la clase positiva
y_probs = rf_model.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calcular el AUC
roc_auc = roc_auc_score(y_test, y_probs)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Línea diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva Característica del Receptor (ROC)')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# ## AdaBoost

# Calcular pesos de clase
class_weights = class_weight.compute_sample_weight('balanced', y_train)

# Crear y ajustar el modelo de AdaBoost
adaboost_model = AdaBoostClassifier(n_estimators=100, random_state=5555, algorithm='SAMME')
adaboost_model.fit(X_train, y_train, sample_weight=class_weights)


# Hacer predicciones
y_pred_adaboost = adaboost_model.predict(X_test)


# Evaluar el modelo
print(classification_report(y_test, y_pred_adaboost))

plt_conf_matrix(y_test,y_pred_adaboost)


# Importancias de las características
importances_adaboost = adaboost_model.feature_importances_
indices = np.argsort(importances_adaboost)[::-1]

# Graficar las importancias de las características
plt.figure()
plt.title('Importancias de las Características')
plt.bar(range(X.shape[1]), importances_adaboost[indices], align='center')
plt.xticks(range(X.shape[1]), np.array(X.columns)[indices],rotation=90,)
plt.xlim([-1, X.shape[1]])
plt.show()


# Obtener las probabilidades predichas para la clase positiva
y_probs = adaboost_model.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calcular el AUC
roc_auc = roc_auc_score(y_test, y_probs)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Línea diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva Característica del Receptor (ROC)')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# ## XGBoost

# Calcular el peso para la clase positiva
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Crear y ajustar el modelo de XGBoost
xgb_model = XGBClassifier(n_estimators=1000, max_depth=10, random_state=5555, scale_pos_weight=scale_pos_weight)
xgb_model.fit(X_train, y_train)

# Hacer predicciones
y_pred_xgboost = xgb_model.predict(X_test)

# Evaluar el modelo
print(classification_report(y_test, y_pred_xgboost))

plt_conf_matrix(y_test,y_pred_xgboost)



# Importancias de las características
importances_xgboost = xgb_model.feature_importances_
indices = np.argsort(importances_xgboost)[::-1]

# Graficar las importancias de las características
plt.figure()
plt.title('Importancias de las Características')
plt.bar(range(X.shape[1]), importances_xgboost[indices], align='center')
plt.xticks(range(X.shape[1]), np.array(X.columns)[indices],rotation=90,)
plt.xlim([-1, X.shape[1]])
plt.show()


# Obtener las probabilidades predichas para la clase positiva
y_probs = xgb_model.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calcular el AUC
roc_auc = roc_auc_score(y_test, y_probs)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Línea diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva Característica del Receptor (ROC)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

