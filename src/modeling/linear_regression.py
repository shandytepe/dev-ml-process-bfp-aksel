from sklearn.linear_model import LinearRegression
from src.utils.helper import dump_joblib
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def modeling_linreg(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    linreg = LinearRegression()

    linreg.fit(X_train, y_train)
    
    dump_joblib(linreg, params["model_dump_path"] + "vanilla_linreg_model.pkl")
    
    return linreg


def predict_baseline(model, X_valid, y_valid):
    y_pred_dummy = model.predict(X_valid)
    
    print(f"MSE: {mean_squared_error(y_valid, y_pred_dummy)}")
    print(f"R2: {r2_score(y_valid, y_pred_dummy)}")
