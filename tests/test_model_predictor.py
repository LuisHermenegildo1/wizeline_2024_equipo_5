import pytest
from unittest.mock import patch, MagicMock
from app.model.model import ModelPredictor

# Usamos fixtures para preparar el entorno antes de cada test
@pytest.fixture
@patch('app.model.model.pickle.load')
@patch('app.model.model.open')
def model_predictor_fixture(mock_open, mock_pickle_load):
    """
    Fixture que simula la carga del modelo y del vectorizador.
    """
    # Crear un mock para el modelo
    mock_model = MagicMock()
    mock_model.predict.return_value = [10.0]  # Simulamos que el modelo siempre predice 10.0

    # Crear un mock para el vectorizador
    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = 'mocked_vector'

    # Simular lo que devuelve pickle.load
    mock_pickle_load.side_effect = [mock_model, mock_vectorizer, {0: 'United Kingdom'}]

    # Inicializar el predictor
    return ModelPredictor()

def test_preprocess_text(model_predictor_fixture):
    """
    Verificar que el preprocesamiento del texto funcione correctamente.
    """
    raw_text = "White metal lantern, price: $20.99"
    expected_output = "white metal lantern price"
    
    # Ejecutar el método preprocess_text
    processed_text = model_predictor_fixture.preprocess_text(raw_text)

    # Comprobar que la salida sea la esperada
    assert processed_text == expected_output

def test_predict(model_predictor_fixture):
    """
    Verificar que el método predict funcione y llame al modelo correctamente.
    """
    description = "white metal lantern"
    
    # Ejecutar el método predict
    predicted_price = model_predictor_fixture.predict(description)

    # Comprobar que el vectorizador se haya llamado correctamente
    model_predictor_fixture.vectorizer.transform.assert_called_once_with([description])

    # Comprobar que el modelo haya realizado la predicción
    model_predictor_fixture.model.predict.assert_called_once_with('mocked_vector')

    # Comprobar que el resultado de la predicción sea el esperado
    assert predicted_price == 10.0