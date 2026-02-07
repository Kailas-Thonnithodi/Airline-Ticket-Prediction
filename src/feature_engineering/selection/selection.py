import pandas as pd
import numpy as np
import statistics
from warnings import filterwarnings

if __name__ == "__main__":
    data_location =  'data/modified/feature_engineered_data.csv'
    data = pd.read_csv(data_location)

    data = data.drop(columns=['Date_of_Journey_Year'], axis=1)
    data.to_csv('data/modified/feature_selection_data.csv', index=False)
    print("Generated a CSV File in modified folder!")
