import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import List
import numpy as np
import math
import scipy


def mean(args) -> float:
    elements_sum: float = 0
    for el in args:
        elements_sum += el
    return float(elements_sum) / len(args)

def standard_deviation(args) -> float:
    counter: float = 0
    denominator: float = len(args) - 1
    args_mean: float = mean(args)

    for el in args:
        temp = (el - args_mean)**2
        counter += temp
    
    return math.sqrt(counter/denominator)


def correlation_coefficient(dataFrame) -> float:
    #  
    dataFrame_copy = pd.DataFrame(dataFrame[:], dtype=float)
    n = len(dataFrame['Y'])

    dataFrame_copy['Y2'] = dataFrame['Y'] ** 2
    dataFrame_copy['X2'] = dataFrame['X'] ** 2
    dataFrame_copy['XY'] = dataFrame['X'] * dataFrame['Y']
    # dodaje nam wiersz o nazwie `sum`
    dataFrame_copy.loc['sum'] = dataFrame_copy.sum()
    counter: float = dataFrame_copy.at['sum', 'XY'] * n
    counter -= dataFrame_copy.at['sum', 'X'] * dataFrame_copy.at['sum', 'Y']
    
    denominator: float = n * dataFrame_copy.at['sum', 'X2'] - (dataFrame_copy.at['sum', 'X'] ** 2)
    denominator *= (n * dataFrame_copy.at['sum', 'Y2'] - (dataFrame_copy.at['sum', 'Y'] ** 2))
    denominator = math.sqrt(denominator)

    return counter/denominator


dataFrame = pd.DataFrame()
dataFrame['X'] = [1, 2, 3, 4, 5]
dataFrame['Y'] = [4, 6, 9, 11, 18]
print(dataFrame)

print("np.mean(dataFrame['X']) == mean(dataFrame['X']):")
print(np.mean(dataFrame['X']) == mean(dataFrame['X']))

print()

print("np.mean(dataFrame['Y']) == mean(dataFrame['Y']):")
print(np.mean(dataFrame['Y']) == mean(dataFrame['Y']))

print()

print(f"np.std(dataFrame['X']) = {np.std(dataFrame['X'])}")
print(f"np.std(dataFrame['Y']) = {np.std(dataFrame['Y'])}")

print()

print(f"standard_deviation(dataFrame['X']) = {standard_deviation(dataFrame['X'])}")
print(f"standard_deviation(dataFrame['Y']) = {standard_deviation(dataFrame['Y'])}")


# dataFrame['Y2'] = dataFrame['Y'] ** 2
# dataFrame['X2'] = dataFrame['X'] ** 2
# dataFrame['xy'] = dataFrame['X'] * dataFrame['Y']
# dataFrame.loc['sum'] = dataFrame.sum()

print(f"my function correlation_coefficient(dataFrame) = {correlation_coefficient(dataFrame)}")

print()

correlation_scipy = scipy.stats.pearsonr(dataFrame['X'], dataFrame['Y'])
print(f"correlation scipy: {correlation_scipy[0]}")

print()

b = standard_deviation(dataFrame['Y']) / standard_deviation(dataFrame['X'])
b *= correlation_coefficient(dataFrame)

a = mean(dataFrame['Y']) - (b * mean(dataFrame['X']))

print(f"b = {b}")
print(f"a = {a}")

def linia_regresji(x):
    return (b * x) + a

# Wykres regresji liniowej:

# x = np.linspace(0, 5, 1000)
# plt.scatter(dataFrame['X'], dataFrame['Y'], label='Wartości Niezależne')
# plt.plot(x, linia_regresji(x), 'r', label='Linia Regresji')
# plt.xlabel('Wartości X')
# plt.ylabel('Wartości Y')
# plt.legend()
# plt.show()

dataFrame = dataFrame._append({'X': 6, 'Y': np.nan}, ignore_index=True)
dataFrame = dataFrame._append({'X': 7, 'Y': np.nan}, ignore_index=True)
dataFrame = dataFrame._append({'X': 8, 'Y': np.nan}, ignore_index=True)

def predict_y(x, b, a):
    return b * x + a

dataFrame.at[5, 'Y'] = predict_y(dataFrame['X'][5], b, a)
dataFrame.at[6, 'Y'] = predict_y(dataFrame['X'][6], b, a)
dataFrame.at[7, 'Y'] = predict_y(dataFrame['X'][7], b, a)
print(dataFrame)






