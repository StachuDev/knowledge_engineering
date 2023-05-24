import pandas as pd
from sklearn.datasets import load_iris
import numpy as np


def knn_algorithm(prepared_dataFrame, query_index, k_number):
    df2 = pd.DataFrame()

    df2["average distance"] = (prepared_dataFrame.iloc[:, 0] - prepared_dataFrame.iloc[query_index, 0]) ** 2
    df2["average distance"] += (prepared_dataFrame.iloc[:, 1] - prepared_dataFrame.iloc[query_index, 1]) ** 2
    df2["average distance"] += (prepared_dataFrame.iloc[:, 2] - prepared_dataFrame.iloc[query_index, 2]) ** 2
    df2["average distance"] += (prepared_dataFrame.iloc[:, 3] - prepared_dataFrame.iloc[query_index, 3]) ** 2
    df2["average distance"] = df2["average distance"] ** (1/2)
    df2["species"] = prepared_dataFrame["species"]
 
    return round(np.mean(df2.sort_values(by=["average distance"]).head(k_number)))


def test_knn_algorithm(dataFrame, k_number, proportions):
    y_test = []
    start_index = int(proportions*100)
    for i in range(start_index, 100):
        output = knn_algorithm(dataFrame, i, k_number)
        y_test.append(output)

    return np.asarray(y_test)


if __name__ == '__main__':

    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    df['species'] = iris.target
    cropped_dataFrame = df.head(100)
    print(df)
    # df.sample(frac = 1) miesza wiersze tabeli
    prepared_dataFrame = cropped_dataFrame.sample(frac=1)
    # prepared_dataFrame.set_index(pd.Index(range(100)))

    # poprawne warosci gatunków zapisane w tablicy 
    series = prepared_dataFrame.iloc[60:100, 4]
    array = series.to_numpy()
    print(array)

    # przewidywane wartosci gatunków wyliczne przez algorytm
    print(test_knn_algorithm(prepared_dataFrame, k_number=5, proportions=0.6))