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
    hospital_data.drop("as_of_date", axis='columns', inplace=True)

    X = hospital_data.drop("COVID_HOSP", axis="columns", inplace=False)
    y = hospital_data.loc[:, "COVID_HOSP"]

    X.drop([
        "numcases_total",
        "numdeaths_weekly",
        "ratedeaths_total",
        "ratecases_last7",
        "ratedeaths_last7",
        "numcases_last14",
        "numdeaths_last14",
        "avgcases_last7",
        "avgincidence_last7",
        "avgratedeaths_last7",
        "numtotal_pfizerbiontech_distributed",
        "numtotal_pfizerbiontech_5_11_distributed",
        "numtotal_moderna_distributed",
        "numtotal_astrazeneca_distributed",
        "numtotal_janssen_distributed",
        "numtotal_novavax_distributed",
    ], axis='columns', inplace=True)

    # 2nd iteration of feature importance
    X.drop([
        "reporting_week",
        "reporting_year",
        "numcases_weekly",
        "ratedeaths_last14",
        "numtotal_all_distributed",
    ], axis='columns', inplace=True)

    return (X, y)


def feature_importance(model, X, y):
    model.fit(X, y)
    importance = model.coef_
    # summarize feature importance

    for i, v in enumerate(importance):
        print('Feature: {}, Score: {:4f}'.format(list(X)[i], v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

    return model


def main():
    X, y = data_preprocessing()
    feature_importance(LinearRegression(), X, y)


main()
