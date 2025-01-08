from sklearn.dummy import DummyRegressor
from src.utils.helper import dump_joblib
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def modeling_baseline(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    dummy_regr = DummyRegressor(strategy = "mean")

    dummy_regr.fit(X_train, y_train)
    
    dump_joblib(dummy_regr, params["model_dump_path"] + "baseline_model.pkl")
    
    return dummy_regr


def predict_baseline(model, X_valid, y_valid):
    y_pred_dummy = model.predict(X_valid)
    
    print(f"MSE: {mean_squared_error(y_valid, y_pred_dummy)}")
    print(f"R2: {r2_score(y_valid, y_pred_dummy)}")
