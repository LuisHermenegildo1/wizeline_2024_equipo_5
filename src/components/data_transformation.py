import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from pathlib import Path

class DataTransformer:
    def __init__(self, stg_folder, prod_folder, prod_file):
        self.stg_folder = Path(stg_folder)
        self.prod_folder = Path(prod_folder)
        self.prod_file = prod_file
        self.preprocessor = None

    def load_data(self):
        try:
            df = pd.read_csv(self.stg_folder / 'OnlineRetail.csv', encoding='ISO-8859-1')
            print("Data loaded successfully")
            return df
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return None

    def clean_data(self, df):
        # Eliminar duplicados
        df.drop_duplicates(inplace=True)

        # Eliminar filas con valores nulos en CustomerID
        df.dropna(subset=['CustomerID'], inplace=True)

        # Convertir InvoiceDate a datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        # Rellenar valores nulos en la columna Description con "Unknown"
        df['Description'].fillna('Unknown', inplace=True)

        # Convertir a minúsculas y eliminar espacios innecesarios en la columna Description
        df['Description'] = df['Description'].str.lower().str.strip()

        # Estandarizar nombres de países
        df['Country'] = df['Country'].str.title()

        # Calcular el dinero gastado por transacción
        df['TotalSpent'] = df['Quantity'] * df['UnitPrice']

        # Agregar campos derivados
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Day'] = df['InvoiceDate'].dt.day
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek

        print("Data cleaned successfully")
        return df

    def fit_transform(self, df_model):
        # Definir características numéricas y categóricas
        numeric_features = ['Quantity']
        categorical_features = ['StockCode']
        
        # Pipelines de transformación
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler(feature_range=(1, 2)))
        ])
        
        cat_pipeline = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first'))
        ])
        
        # ColumnTransformer para procesar numéricas y categóricas
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, numeric_features),
                ('cat', cat_pipeline, categorical_features)
            ],
            remainder='passthrough'
        )

        # Aplicar las transformaciones
        X_transformed = self.preprocessor.fit_transform(df_model)
        return X_transformed

    def transform(self, df_model):
        # Aplicar las transformaciones ya ajustadas
        return self.preprocessor.transform(df_model)

    def save_clean_data(self, df):
        self.prod_folder.mkdir(parents=True, exist_ok=True)  # Crear el directorio si no existe
        df.to_csv(self.prod_folder / self.prod_file, index=False)
        print(f"Cleaned data saved to {self.prod_folder / self.prod_file}")

if __name__ == "__main__":
    transformer = DataTransformer(
        stg_folder='data/stagging_data',
        prod_folder='data/production_data',
        prod_file='OnlineRetail_clean.csv'
    )

    df = transformer.load_data()
    if df is not None:
        clean_df = transformer.clean_data(df)
        X_transformed = transformer.fit_transform(clean_df)
        transformer.save_clean_data(clean_df)