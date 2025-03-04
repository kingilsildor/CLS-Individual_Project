import pandas as pd
from script.flood_analysis import create_flood_level, flood_analysis
from src.data_exportation import export_data
from src.data_importation import transform_datetime


def main():
    # # Setup the data
    # flood_df = load_data()
    # coords_dict = get_coords(flood_df, 2)
    # flood_df = add_coords_to_df(flood_df, coords_dict, 2)
    # flood_df = transform_datetime(flood_df, "Start_Date")
    # flood_df = transform_datetime(flood_df, "End_Date")

    # flood_df = categorize_floods(flood_df, 4)
    # export_data(flood_df, "flood_data.csv")

    df = pd.read_csv("data/flood_data.csv")
    df = create_flood_level(df, admin_level=2, nbins=5)
    transform_datetime(df, "Start_Date")
    transform_datetime(df, "End_Date")

    data = flood_analysis(df, admin_level=2)
    export_data(data, "flood_analysis.csv")
    print(data)


if __name__ == "__main__":
    main()
