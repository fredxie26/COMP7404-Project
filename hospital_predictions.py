import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from mlxtend.evaluate import bias_variance_decomp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

# CONSTANTS
HOSPITAL_DATASET_FILENAME = "combined-dataset.csv"
TRAIN_RATIO = 0.75
RANDOM_STATE = 42


# Data Proprecessing goes here
# Returns a tuple (X,y)
# with X being a data frame with features used for input and y being the output
def data_preprocessing():
    hospital_data = pd.read_csv(HOSPITAL_DATASET_FILENAME)
    # removing dates
    hospital_data.drop("date", axis='columns', inplace=True)
    hospital_data.drop("as_of_date", axis='columns', inplace=True)
    hospital_data.drop("reporting_week", axis='columns', inplace=True)
    hospital_data.drop("reporting_year", axis='columns', inplace=True)

    X = hospital_data.drop("COVID_HOSP", axis="columns", inplace=False)
    y = hospital_data.loc[:, "COVID_HOSP"]

    #X.drop(['numdeaths_weekly', 'avgratedeaths_last7', 'numtotal_janssen_distributed', 'numtotal_novavax_distributed'], axis='columns', inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    

    return (X, y)


def score_regression(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model.score(X_test, y_test)


def predict_regression(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


def main():
    X, y = data_preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
    print("Linear Regression\nCoefficient of Determination: ", score_regression(LinearRegression(), X_train, y_train, X_test, y_test))
    mse, bias, var = bias_variance_decomp(LinearRegression(), X_train, y_train.values, X_test, y_test.values, loss='mse', num_rounds=200, random_seed=RANDOM_STATE)
    print('MSE: %.3f' % mse)
    print('Bias: %.3f' % bias)
    print('Variance: %.3f\n' % var)
    
    print("Ridge Regression\nCoefficient of Determination: ", score_regression(Ridge(), X_train, y_train, X_test, y_test))
    mse, bias, var = bias_variance_decomp(Ridge(), X_train, y_train.values, X_test, y_test.values, loss='mse', num_rounds=200, random_seed=RANDOM_STATE)
    print('MSE: %.3f' % mse)
    print('Bias: %.3f' % bias)
    print('Variance: %.3f\n' % var)

    X, y = data_preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
    print("Decision Tree Regression\nCoefficient of Determination: ", score_regression(DecisionTreeRegressor(max_depth=2), X_train, y_train, X_test, y_test))
    mse, bias, var = bias_variance_decomp(DecisionTreeRegressor(max_depth=2), X_train, y_train.values, X_test, y_test.values, loss='mse', num_rounds=200, random_seed=RANDOM_STATE)
    print('MSE: %.3f' % mse)
    print('Bias: %.3f' % bias)
    print('Variance: %.3f\n' % var)
    
    
    print("Multi-layer Perceptron Regression\nCoefficient of Determination: ", score_regression(MLPRegressor(random_state=RANDOM_STATE, max_iter=5000), X_train, y_train, X_test, y_test))
    #mse, bias, var = bias_variance_decomp(MLPRegressor(random_state=RANDOM_STATE, max_iter=5000), X_train, y_train.values, X_test, y_test.values, loss='mse', num_rounds=200, random_seed=RANDOM_STATE)
    #print('MSE: %.3f' % mse)
    #print('Bias: %.3f' % bias)
    #print('Variance: %.3f\n' % var)
    



if __name__ == "__main__":
    main()
