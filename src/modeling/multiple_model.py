from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso
from src.utils.helper import dump_joblib
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def modeling_multiple(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    dt_baseline = DecisionTreeRegressor()
    ridge_baseline = Ridge()
    lasso_baseline = Lasso()
    
    dt_baseline.fit(X_train, y_train)
    ridge_baseline.fit(X_train, y_train)
    lasso_baseline.fit(X_train, y_train)
    
    dump_joblib(dt_baseline, params["model_dump_path"] + "dt_baseline.pkl")
    dump_joblib(ridge_baseline, params["model_dump_path"] + "dt_baseline.pkl")
    dump_joblib(lasso_baseline, params["model_dump_path"] + "dt_baseline.pkl")

    return dt_baseline, ridge_baseline, lasso_baseline


# def modeling_linreg(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
#     linreg = LinearRegression()

#     linreg.fit(X_train, y_train)
    
#     dump_joblib(linreg, params["model_dump_path"] + "vanilla_linreg_model.pkl")
    
#     return linreg


# def predict_baseline(model, X_valid, y_valid):
#     y_pred_dummy = model.predict(X_valid)
    
#     print(f"MSE: {mean_squared_error(y_valid, y_pred_dummy)}")
#     print(f"R2: {r2_score(y_valid, y_pred_dummy)}")
