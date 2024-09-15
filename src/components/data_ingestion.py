import pandas as pd

def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[df['Quantity'] > 0]  # Filtrar ventas
    df = df[df['UnitPrice'] > 0]  # Filtrar ofertas/promociones
    df = df[df['Country'] == 'United Kingdom']  # Filtrar por país
    df = df[df['StockCode'].isin(['85123A'])]  # Filtrar por StockCode
    
    # Eliminar columnas innecesarias
    df_clean = df.drop(columns=['InvoiceNo', 'Country', 'Description', 'InvoiceDate'])
    
    return df_clean

def check_nulls(df):
    """Verifica si hay datos nulos en el DataFrame."""
    return df.isnull().sum()
