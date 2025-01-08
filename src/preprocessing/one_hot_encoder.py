import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib


def ohe_fit(data_train: pd.DataFrame, column: str, params: dict):
    ohe = OneHotEncoder(categories = [params["value_furnish_status"]], sparse_output=False)

    ohe.fit(data_train[[column]])
    
    joblib.dump(ohe, params["dataset_dump_path"]["processed"] + "ohe_fix.pkl")
    
    return ohe


def preprocess_ohe(data: pd.DataFrame, column: str, ohe) -> pd.DataFrame:
    ohe_feat = ohe.transform(data[[column]])
    
    # create dataframe
    ohe_cols = ohe.categories_[0]
    ohe_df = pd.DataFrame(ohe_feat, columns = ohe_cols, index = data.index)
    
    final_df = pd.concat([data, ohe_df], axis = 1)

    final_df = final_df.drop(columns = [column])
    
    return final_df
