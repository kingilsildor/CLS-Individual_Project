import pandas as pd
from src.assert_statements import assert_df


def export_data(df: pd.DataFrame, file_name: str = "data/flood_data.csv") -> None:
    """
    Export the given data to a CSV file.

    Params
    ------
    - df (pd.DataFrame): The data to export.
    - file_name (str): The name of the file to export the data to. Default is "data/flood_data.csv".
    """
    assert_df(df)
    assert isinstance(file_name, str), "File name is not a string."

    df.to_csv(file_name, index=False)
    assert pd.read_csv(file_name).shape == df.shape, (
        "Data was not exported to the file correctly."
    )
