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


HOSPITAL_DATASET_FILENAME = "combined-dataset.csv"

def data_preprocessing():
    hospital_data = pd.read_csv(HOSPITAL_DATASET_FILENAME)

    # removing dates
    hospital_data.drop("date", axis='columns', inplace=True)

    X = hospital_data.drop("COVID_HOSP", axis="columns", inplace=False)
    y = hospital_data.loc[:, "COVID_HOSP"]

    enc = OneHotEncoder()
    X = enc.fit_transform(X)
    return (X, y)


def feature_importance(model, X, y):
    model.fit(X, y)
    importance = model.coef_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()
    return model


def main():
    X, y = data_preprocessing()
    feature_importance(LinearRegression(), X, y)


main()