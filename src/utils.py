import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot(df, column):
    """Genera un diagrama de cajas para una columna."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=column, data=df)
    plt.title(f'Diagrama de Cajas para {column}')
    plt.ylabel(column)
    plt.show()
