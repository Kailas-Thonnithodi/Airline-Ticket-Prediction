import pandas as pd
import numpy as np
import statistics
from warnings import filterwarnings

def target_guided_encoding(data: pd.DataFrame, column: str): 
    sorted_mean_column = data.groupby([column])['Price'].mean().sort_values().index
    key_pair_column = {key: index for index, key in enumerate(sorted_mean_column, 0)}
    data[column] = data[column].map(key_pair_column)

def label_encoding(data, column):
    unique_values = list(data[column].unique())
    mapping = {category: i for i, category in enumerate(unique_values,0)}
    data[column] = data[column].map(mapping)

if __name__ == "__main__":

    # Loading Data
    filterwarnings("ignore")
    cleaned_data = "data/modified/cleaned_data.csv"
    data = pd.read_csv(cleaned_data)

    # Changing New-Delhi instances to Delhi
    data['Destination'].replace('New Delhi','Delhi',inplace=True)

    # Applying targeted guided encoding on the following columns
    target_guided_encoding(data, 'Airline')
    target_guided_encoding(data, 'Source')
    target_guided_encoding(data, 'Destination')

    # Applying label encoding on the following column
    label_encoding(data, 'Total_Stops')
    label_encoding(data, 'Additional_Info')

    # Dropping column due to too many unique values
    data.drop(columns=["Route"], axis=1, inplace=True)

    # applying boxplot iqr on outliers in price
    quantile1 = data['Price'].quantile(0.25)
    quantile3 = data['Price'].quantile(0.75)
    interquartile_range = quantile3 - quantile1

    # This will indicate all data in the outlier range (check out IQR or boxplot)
    data['Price'] = np.where(data['Price'] >= data['Price'].median(), data['Price'].median(), data['Price'])

    data.to_csv('data/modified/feature_engineered_data.csv', index=False)
    print("Generated a CSV File in modified folder!")
