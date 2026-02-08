import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

def feature_target_extraction(data: pd.DataFrame, target):
    X = data.drop(columns=[target], axis=1)
    y = data[target]
    return X, y

def model_result(model, X_train, X_test, y_train, y_test, generate_model: bool = False):

    # Model Fitting
    model = model.fit(X_train, y_train)
    print(f'\n{type(model).__name__}')
    print('Training Score: {}'.format(model.score(X_train, y_train)))
    
    # Model Predictions
    y_prediction = model.predict(X_test)
    print('Predictions are: {}'.format(y_prediction))

    # Scores
    r2_score = metrics.r2_score(y_test, y_prediction)
    mse = metrics.mean_absolute_error(y_test, y_prediction)
    rmse = metrics.root_mean_squared_error(y_test, y_prediction)
    mae = metrics.mean_squared_error(y_test, y_prediction)
    mape = metrics.mean_absolute_percentage_error(y_test, y_prediction)
    print('R2: {}'.format(r2_score))
    print('MSE: {}'.format(mse))
    print('RMSE: {}'.format(rmse))
    print('MAE: {}'.format(mae))
    print('MAPE: {}%'.format(mape))

    if generate_model == True:
        model_dir = open(f'src/models/{type(model).__name__}_model.pkl', 'wb')
        pickle.dump(model, model_dir)


    

if __name__ == "__main__":
    
    # Original Data
    data = pd.read_csv('data/modified/feature_selection_data.csv')
    X, y = feature_target_extraction(data, 'Price')
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model fit, prediction and the evaluation
    rfr = RandomForestRegressor()
    dtr = DecisionTreeRegressor()
    gbr = GradientBoostingRegressor()
    model_result(rfr, X_train, X_test, y_train, y_test, generate_model=True)
    model_result(dtr, X_train, X_test, y_train, y_test, generate_model=True)
    model_result(gbr, X_train, X_test, y_train, y_test, generate_model=True)

