from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    """Realiza la transformación de los datos numéricos y categóricos."""
    
    numeric_features = ['Quantity']
    categorical_features = ['StockCode']
    
    # Pipeline para datos numéricos
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler(feature_range=(1, 2)))
    ])
    
    # Pipeline para datos categóricos
    cat_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))
    ])
    
    # Column Transformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ], remainder='passthrough')
    
    # Separar características y etiquetas
    X = df[['Quantity']]
    y = df['Quantity']
    
    X_transformed = preprocessor.fit_transform(X)
    
    return X_transformed, y
