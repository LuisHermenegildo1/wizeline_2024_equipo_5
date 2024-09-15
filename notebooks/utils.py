import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Generamos funcion para visualizar si tienen datos vacios
def hasNullData(_df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    sns.heatmap(_df.isnull(), cbar=False, cmap='viridis')
    plt.title('Mapa de Calor de Valores Nulos (NaN)')
    plt.show()