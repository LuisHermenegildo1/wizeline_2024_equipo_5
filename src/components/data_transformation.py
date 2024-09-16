#Instalar requerimientos
#!pip install pandas
#!pip install matplotlib
#!pip install seaborn
#!pip install scikit-learn

#Importamos Librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#project variables
l_stg_folder = 'data/stagging_data'
l_prod_folder = 'data/production_data'
l_prod_file = 'OnlineRetail_clean.csv'

# Cargar los datos
df = pd.read_csv(f'./{l_stg_folder}/OnlineRetail.csv', encoding='ISO-8859-1')

##Limpiea de los datos

# Eliminar duplicados
df.drop_duplicates(inplace=True)

# Eliminar filas con valores nulos en CustomerID
df.dropna(subset=['CustomerID'], inplace=True)

# Convertir InvoiceDate a datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Rellenar valores nulos en la columna Description con "Unknown"
df['Description'].fillna('Unknown', inplace=True)

# Eliminar filas con cantidad negativa o cero
# Se omite registro debido a que son devoluciones
#df = df[df['Quantity'] > 0]

# Eliminar filas con precio negativo o cero
# Se omite debido a que puede ser una oferta o devolución
#df = df[df['UnitPrice'] > 0]

# Convertir a minúsculas y eliminar espacios innecesarios en la columna Description
df['Description'] = df['Description'].str.lower().str.strip()

# Estandarizar nombres de países
df['Country'] = df['Country'].str.title()

# Calcular el dinero gastado por transacción
df['TotalSpent'] = df['Quantity'] * df['UnitPrice']

# Calculo de campo year
df['Year'] = df['InvoiceDate'].dt.year

# Calculo de campo month

df['Month'] = df['InvoiceDate'].dt.month

# Calculo de campo day

df['Day'] = df['InvoiceDate'].dt.day

# Calculo de campo hour
df['Hour'] = df['InvoiceDate'].dt.hour

# Calculo de campo dayOfWeek

df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek


#Save Clean Data Set

df.to_csv(f'./{l_prod_folder}/{l_prod_file}', index=False)