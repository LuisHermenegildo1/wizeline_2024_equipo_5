from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Support Vector Regressor': SVR(),
            'Random Forest Regressor': RandomForestRegressor()
        }

        self.param_grids = {
            'Linear Regression': {
                'fit_intercept': [True, False]
            },
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

        self.random_search_grids = {
            'Linear Regression': {
                'fit_intercept': [True, False]
            },
            'Support Vector Regressor': {
                'kernel': ['linear', 'rbf'],  
                'C': [0.01, 0.1, 1, 10],  
                'epsilon': [0.1, 0.01, 0.001],
                'gamma': ['scale', 'auto']
            },
            'Random Forest Regressor': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    
    def grid_search(self, X, y):
        best_models = {}
        model_scores = {}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for model_name in self.models:
            model = self.models[model_name]
            param_grid = self.param_grids[model_name]

            start_time = time.time()

            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            duration = time.time() - start_time

            best_model = grid_search.best_estimator_
            y_pred = grid_search.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            best_models[model_name] = best_model
            model_scores[model_name] = {'mse': mse, 'r2': r2, 'mae': mae}

            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Test MSE: {mse}, R-squared: {r2}, MAE: {mae}")
            print(f"Training time: {duration:.2f} seconds\n")

        return best_models, model_scores

    def random_search(self, X, y):
        best_models = {}
        model_scores = {}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for model_name in self.models:
            model = self.models[model_name]
            param_dist = self.random_search_grids[model_name]

            start_time = time.time()

            random_search = RandomizedSearchCV(model, param_distributions=param_dist, cv=5, 
                                               scoring='neg_mean_squared_error', n_jobs=-1, n_iter=10, random_state=42)
            random_search.fit(X_train, y_train)

            duration = time.time() - start_time

            best_model = random_search.best_estimator_
            y_pred = random_search.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            best_models[model_name] = best_model
            model_scores[model_name] = {'mse': mse, 'r2': r2, 'mae': mae}

            print(f"Best parameters for {model_name}: {random_search.best_params_}")
            print(f"Test MSE: {mse}, R-squared: {r2}, MAE: {mae}")
            print(f"Training time: {duration:.2f} seconds\n")

        return best_models, model_scores
