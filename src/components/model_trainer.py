from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(estimator, X, y, title):
    """Genera una curva de aprendizaje para un estimador dado."""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Negative MSE")
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def regression_gridsearch(X, y, models, param_grids, metric='mse'):
    """Realiza búsqueda de hiperparámetros con GridSearchCV para varios modelos."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_models = {}
    model_scores = {}

    for model_name in models:
        print(f"Running GridSearchCV for {model_name}...")
        model = models[model_name]
        param_grid = param_grids[model_name]

        start_time = time.time()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        end_time = time.time()

        best_models[model_name] = grid_search.best_estimator_
        y_pred = grid_search.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        model_scores[model_name] = {'mse': mse, 'r2': r2, 'mae': mae}

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Test MSE for {model_name}: {mse}")
        plot_learning_curve(best_models[model_name], X, y, f"Learning Curve: {model_name}")
    
    # Selección del mejor modelo según la métrica
    if metric == 'mse':
        best_model = min(model_scores, key=lambda x: model_scores[x]['mse'])
    elif metric == 'r2':
        best_model = max(model_scores, key=lambda x: model_scores[x]['r2'])
    elif metric == 'mae':
        best_model = min(model_scores, key=lambda x: model_scores[x]['mae'])
    
    return best_models[best_model], model_scores[best_model]
