import pytest
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.components.data_ingestion import load_data, check_nulls
from src.components.data_transformation import preprocess_data
from src.components.model_trainer import regression_gridsearch

# Definir una fixture para cargar los datos
@pytest.fixture
def data():
    """Fixture que carga los datos del archivo CSV."""
    df = load_data('../data/online_retail_train.csv')
    return df

# Fixture para preprocesar los datos
@pytest.fixture
def preprocessed_data(data):
    """Fixture que preprocesa los datos y devuelve X e y."""
    X, y = preprocess_data(data)
    return X, y

# Test para data_ingestion.py
def test_load_data(data):
    """Prueba que el archivo CSV se cargue correctamente y filtre los datos."""
    # Verificamos que el DataFrame no esté vacío
    assert not data.empty, "El DataFrame no debería estar vacío"
    # Verificamos que las columnas filtradas ya no estén en el DataFrame
    assert 'InvoiceNo' not in data.columns, "La columna 'InvoiceNo' debería haber sido eliminada"
    assert 'Country' not in data.columns, "La columna 'Country' debería haber sido eliminada"
    assert 'Description' not in data.columns, "La columna 'Description' debería haber sido eliminada"

def test_check_nulls(data):
    """Prueba que no haya valores nulos después de la limpieza."""
    nulls = check_nulls(data)
    assert nulls.sum() == 0, "No debería haber valores nulos en el DataFrame"

# Test para data_transformation.py
def test_preprocess_data(preprocessed_data):
    """Prueba que los datos se preprocesen correctamente."""
    X, y = preprocessed_data
    # Verificamos que X y y no estén vacíos
    assert X.shape[0] > 0, "El conjunto de características X no debería estar vacío"
    assert y.shape[0] > 0, "El conjunto de etiquetas y no debería estar vacío"

# Test para model_trainer.py
def test_regression_gridsearch(preprocessed_data):
    """Prueba el flujo de GridSearchCV para asegurar que se entrenan y seleccionan modelos correctamente."""
    X, y = preprocessed_data
    
    # Definir un modelo simple y su parámetro
    models = {
        'Linear Regression': LinearRegression()
    }

    param_grids = {
        'Linear Regression': {'fit_intercept': [True, False]}
    }

    best_model, metrics = regression_gridsearch(X, y, models, param_grids)
    
    # Verificamos que el mejor modelo sea instanciado correctamente
    assert isinstance(best_model, LinearRegression), "El mejor modelo debería ser una instancia de LinearRegression"
    
    # Verificamos que las métricas sean razonables
    assert metrics['mse'] >= 0, "El error cuadrático medio (MSE) debería ser un valor positivo"
