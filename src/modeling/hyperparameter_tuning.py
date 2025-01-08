from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from src.utils.helper import load_joblib, dump_joblib


def hyperparam_process(model_path: str, X_train: pd.DataFrame, y_train: pd.Series):
    model = load_joblib(path = model_path)
    
    PARAMS_LASSO = {
        "alpha": np.arange(0.00, 1.0, 0.01),
        "max_iter": np.arange(1000, 10_000, 250)
    }
    
    k_folds = KFold(n_splits = 5)
    
    best_lasso_random = RandomizedSearchCV(estimator = model,
                                           param_distributions = PARAMS_LASSO,
                                           cv = k_folds,
                                           verbose = 3)
    
    best_lasso_random.fit(X_train, y_train)
    
    return best_lasso_random.best_params_


def best_model_train(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    best_model = Lasso(alpha = 0.98, max_iter = 5500)
    
    best_model.fit(X_train, y_train)
    
    dump_joblib(best_model, params["model_dump_path"] + "best_model.pkl")
    
    return best_model


def predict_best_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R2: {r2_score(y_test, y_pred)}")
