import pandas as pd


def custom_label_encoder(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    MAPPER_VALUE = {
        "no": 0,
        "yes": 1
    }
    
    for col in params["label_encoder_columns"]:
        data[col] = data[col].replace(MAPPER_VALUE)
        
    return data
