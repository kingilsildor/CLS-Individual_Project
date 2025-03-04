import pandas as pd

def assert_df(df: pd.DataFrame) -> None:
    """
    Assert that the given data is a pandas DataFrame and has rows and columns.

    Params
    ------
    - df (pd.DataFrame): The data to assert.
    """
    assert isinstance(df, pd.DataFrame), "Data is not a pandas DataFrame."
    assert df.shape[0] > 0, "No data found in the given directory."
    assert df.shape[1] > 0, "No columns found in the given directory."
