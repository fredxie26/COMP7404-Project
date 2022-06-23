import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# CONSTANTS
HOSPITAL_DATASET_FILENAME = "covid19-epiSummary-hospVentICU.csv"
TRAIN_RATIO = 0.75
RANDOM_STATE = 42


# Data Proprecessing goes here
# Returns a tuple (X,y)
# with X being a data frame with features used for input and y being the output
def data_preprocessing():
    hospital_data = pd.read_csv(HOSPITAL_DATASET_FILENAME)

    # removing dates
    hospital_data.drop("Date", axis='columns', inplace=True)

    X = hospital_data.drop("COVID_HOSP", axis="columns", inplace=False)
    y = hospital_data.loc[:, "COVID_HOSP"]
    return (X, y)


def score_regression(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model.score(X_test, y_test)


def main():
    X, y = data_preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
    print("Linear Regression Coefficient of Determination: ", score_regression(LinearRegression(), X_train, y_train, X_test, y_test))
    print("Multi-layer Perceptron Regression Coefficient of Determination: ", score_regression(MLPRegressor(random_state=RANDOM_STATE, max_iter=500), X_train, y_train, X_test, y_test))


if __name__ == "__main__":
    main()
