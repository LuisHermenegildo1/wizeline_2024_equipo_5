import pytest
from unittest.mock import patch, MagicMock
from src.components.model_trainer import ModelTrainer

@pytest.fixture
def model_trainer_fixture():
    """
    Fixture para inicializar la clase ModelTrainer.
    """
    return ModelTrainer()

@patch('src.components.model_trainer.GridSearchCV')
def test_grid_search(mock_grid_search, model_trainer_fixture):
    """
    Verificar que GridSearchCV se llama correctamente y que el mejor modelo se devuelve.
    """
    # Crear un mock para GridSearchCV
    mock_grid_search_instance = MagicMock()
    mock_grid_search.return_value = mock_grid_search_instance
    
    # Simular el mejor modelo y los resultados del ajuste
    mock_grid_search_instance.best_estimator_ = 'best_model'
    mock_grid_search_instance.predict.return_value = [10.0]
    mock_grid_search_instance.best_params_ = {'param': 'value'}
    mock_grid_search_instance.fit.return_value = None

    # Simular datos de entrada
    X = [[1, 2], [3, 4], [5, 6]]
    y = [10, 20, 30]

    # Ejecutar el grid_search en ModelTrainer
    best_models, model_scores = model_trainer_fixture.grid_search(X, y)

    # Verificar que GridSearchCV fue llamado
    mock_grid_search.assert_called()

    # Verificar que el mejor modelo fue guardado correctamente
    assert best_models['Linear Regression'] == 'best_model'
    assert 'mse' in model_scores['Linear Regression']

@patch('src.components.model_trainer.RandomizedSearchCV')
def test_random_search(mock_random_search, model_trainer_fixture):
    """
    Verificar que RandomizedSearchCV se llama correctamente y que el mejor modelo se devuelve.
    """
    # Crear un mock para RandomizedSearchCV
    mock_random_search_instance = MagicMock()
    mock_random_search.return_value = mock_random_search_instance
    
    # Simular el mejor modelo y los resultados del ajuste
    mock_random_search_instance.best_estimator_ = 'best_model_random'
    mock_random_search_instance.predict.return_value = [15.0]
    mock_random_search_instance.best_params_ = {'param': 'random_value'}
    mock_random_search_instance.fit.return_value = None

    # Simular datos de entrada
    X = [[1, 2], [3, 4], [5, 6]]
    y = [15, 25, 35]

    # Ejecutar el random_search en ModelTrainer
    best_models, model_scores = model_trainer_fixture.random_search(X, y)

    # Verificar que RandomizedSearchCV fue llamado
    mock_random_search.assert_called()

    # Verificar que el mejor modelo fue guardado correctamente
    assert best_models['Linear Regression'] == 'best_model_random'
    assert 'mse' in model_scores['Linear Regression']