import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings

def change_column_to_datetime(column):
    data[column] = pd.to_datetime(data[column])

def day_month_year(data, column):
    if column not in  data.columns:
        return data
    else:
        data[f"{column}_Day"] = data[column].dt.day
        data[f"{column}_Month"] = data[column].dt.month
        data[f"{column}_Year"] = data[column].dt.year
        data = data.drop(columns=[column])
    return data

def hour_minute(data, column):
    if column not in data.columns:
        return data
    else:
        data[f"{column}_Hour"] = data[column].dt.hour
        data[f"{column}_Minute"] = data[column].dt.minute
        data = data.drop(columns=[column])
    return data

if __name__ == "__main__":

    # Loading Data
    filterwarnings("ignore")
    raw_data = "data/raw_data/raw_data.xlsx"
    data = pd.read_excel(raw_data)

    # Data Cleaning and Preprocessing
    ## remove null rows
    data.dropna(inplace=True)
    ## change datetime columns
    for feature in ['Date_of_Journey','Arrival_Time','Dep_Time']:
        change_column_to_datetime(feature)
    ## create new columns for journey feature
    data = day_month_year(data,"Date_of_Journey")
    data = hour_minute(data, "Dep_Time")
    data = hour_minute(data, "Arrival_Time")

    data.to_csv('data/cleaned_data/cleaned_data.csv', index=False)
    print("Generated a CSV File in cleaned_data folder!")