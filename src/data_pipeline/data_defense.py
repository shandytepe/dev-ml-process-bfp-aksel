import pandas as pd


def data_defense_checker(input_data: pd.DataFrame, params: dict) -> None:
    # try:
    print("===== Start Data Defense Checker =====")
    # check data types
    assert input_data.select_dtypes("object").columns.to_list() == params["object_columns"], "an error occurs in object columns"
    assert input_data.select_dtypes("int").columns.to_list() == params["int64_columns"], "an error occurs in integer columns"
    
    # check values
    assert set(input_data.mainroad).issubset(set(params["value_status"])), "an error occurs on mainroad column"
    assert set(input_data.guestroom).issubset(set(params["value_status"])), "an error occurs on guestroom column"
    assert set(input_data.basement).issubset(set(params["value_status"])), "an error occurs on basement column"
    assert set(input_data.hotwaterheating).issubset(set(params["value_status"])), "an error occurs on hotwaterheating column"
    assert set(input_data.airconditioning).issubset(set(params["value_status"])), "an error occurs on airconditioning column"
    assert set(input_data.prefarea).issubset(set(params["value_status"])), "an error occurs on prefarea column"
    assert set(input_data.furnishingstatus).issubset(set(params["value_furnish_status"])), "an error occurs on furnishingstatus column"

# except Exception:
#     raise Exception("Failed Data Defense Checker")

# finally:
    print("===== Finish Data Defense Checker =====")
