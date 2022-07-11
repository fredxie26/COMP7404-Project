import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HOSPITAL_DATASET_FILENAME = "combined-dataset.csv"
FEATURE_THRESHOLD = 0 #list all features below this threshold

column_names = []

def data_preprocessing():
    hospital_data = pd.read_csv(HOSPITAL_DATASET_FILENAME)
    # removing dates
    hospital_data.drop("date", axis='columns', inplace=True)
    hospital_data.drop("as_of_date", axis='columns', inplace=True)
    hospital_data.drop("reporting_week", axis='columns', inplace=True)
    hospital_data.drop("reporting_year", axis='columns', inplace=True)

    X = hospital_data.drop("COVID_HOSP", axis="columns", inplace=False)
    y = hospital_data.loc[:, "COVID_HOSP"]
    global column_names
    column_names = list(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    

    return (X, y)


def feature_importance(model, X, y):
    model.fit(X, y)
    coeficients = model.coef_
    # summarize feature importance
    unimportant_features = []
    for i, v in enumerate(coeficients):
        print('Feature: {}, Coefficient: {:4f}'.format(column_names[i], v))
        if (abs(v) < FEATURE_THRESHOLD):
            unimportant_features.append(column_names[i])
    # plot feature importance
    print(unimportant_features)
    plt.bar([x for x in range(len(coeficients))], coeficients)
    plt.show()

    return model


def main():
    X, y = data_preprocessing()
    feature_importance(LinearRegression(), X, y)


main()
