import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

if __name__ == "__main__":
    
    # Loading data
    data_dir = 'data/modified/feature_selection_data.csv'
    data = pd.read_csv(data_dir)

    # Create X and y
    X = data.drop('Price', axis=1)
    y = data['Price']

    # train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    # Model Creation (Model predictions are in respective ipynb file)
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    
    # Storing Model
    model_dir = open('src/models/rfr_model.pkl', 'wb')
    pickle.dump(rfr, model_dir)