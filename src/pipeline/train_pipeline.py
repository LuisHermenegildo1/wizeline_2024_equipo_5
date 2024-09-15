import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.components.data_ingestion import load_data
from src.components.data_transformation import preprocess_data
from src.components.model_trainer import regression_gridsearch
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

def train():
    # Ruta absoluta al archivo CSV
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/online_retail_train.csv'))

    # Cargar datos
    df = load_data(file_path)
    
    # Preprocesar datos
    X, y = preprocess_data(df)
    
    # Definir modelos y sus hiperparámetros
    models = {
        'Linear Regression': LinearRegression(),
        'Support Vector Regressor': SVR(),
        'Random Forest Regressor': RandomForestRegressor()
    }

    param_grids = {
        'Linear Regression': {'fit_intercept': [True, False]},
        'Support Vector Regressor': {
            'kernel': ['linear', 'rbf'],
            'C': [0.01, 0.1, 1],
            'epsilon': [0.1, 0.01],
            'gamma': ['scale']
        },
        'Random Forest Regressor': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    }
    
    # Entrenar el modelo
    best_model, metrics = regression_gridsearch(X, y, models, param_grids)
    print(f"Mejor modelo: {best_model}")
    print(f"Métricas: {metrics}")

if __name__ == '__main__':
    train()
