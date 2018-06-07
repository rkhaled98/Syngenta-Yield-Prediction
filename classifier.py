import pandas as pd
import numpy as np


df = pd.read_csv('Syngenta/Syngenta_2017/Experiment_dataset.csv')

# print(df.describe())
# print(df.head())

feature_columns = ['Organic matter', 'pH', 'Clay', 'Silt']

X = df.loc[:, feature_columns]
# print(X.head())
# print(X.shape)

y = df.Yield

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.head())
# print(X_test.head())
print(y_train.head())
print(X_train.shape)
print(y_train.shape)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

classifier = LinearDiscriminantAnalysis()

print()
classifier.fit(X_train, y_train)

# y = 

# X_train, X_test, y_train, y_test = train_test_split()

