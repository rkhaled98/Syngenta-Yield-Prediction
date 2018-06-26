# THERE ARE SEPARATE NOTEBOOKS FOR VISUALIZATIONS, DATASET ANALYSIS, ETC. IN THE REPO.

import pandas as pd
import numpy as np

# READ THE CSV INTO DATAFRAME

df = pd.read_csv('Syngenta/Syngenta_2017/Experiment_dataset.csv')

# df_sf = pd.read_csv('Smart_Farm_dataset.csv')


# CURRENTLY NECESSARY IF: USING 174 ADDITIONAL VARIETY COLUMNS METHOD

# THIS IS A DIFFERENT APPROACH TO THE ABOVE FOUR CELLS, WHERE WE HAVE 174 ADDITIONAL FEATURE COLUMNS
# EACH WITH A 0 (IF IT IS NOT OF THAT VARIETY) OR A 1 (IF IT IS OF THAT VARIETY)

# variety_dummies = pd.get_dummies(df.Variety)
# df = pd.concat([df, variety_dummies], axis=1)


# LATITUDE AND LONGITUDE CLUSTERING INTO FEATURES

from sklearn.cluster import KMeans

latlong = df.loc[:, ['Latitude', 'Longitude']]

kmeans = KMeans(n_clusters=4, random_state=0).fit(latlong)
kmeans.labels_.shape
lat_long_dummies = pd.get_dummies(kmeans.labels_)
lat_long_dummies = lat_long_dummies.rename(index=int, columns={0: "Loc Clust 0",
                                                               1: "Loc Clust 1",
                                                               2: "Loc Clust 2",
                                                               3: "Loc Clust 3"})
df = pd.concat([df, lat_long_dummies], axis = 1)

#REMOVE ANY NAN VALUES

print(df.columns)
df = df[~df.Silt.isnull()]
df = df[~df['Loc Clust 1'].isnull()]

# DROP ALL THE CELLS THAT ARE NOT USABLE SUCH AS THE ONES THAT ARE STRINGS OR DATES

# set if want to drop some columns specifically
should_drop = 1
columns_to_drop = ['Experiment', 'Location',
                   'Check Yield', 'Yield difference', 'Latitude',
                   'Longitude', 'Variety', 'PI', 'Planting date']

# set if want to keep some columns specifically
should_keep = 0
# columns_to_keep = ['Loc Clust 0', 'Loc Clust 1', 'Loc Clust 2', 'Loc Clust 3']
columns_to_keep_top = ['Silt', 'Precipitation', 'Temperature', 'Solar Radiation', 'Organic matter']
columns_VARIETIES_ONLY = np.asarray(df.iloc[:, df.columns.str.match('V\d\d\d\d\d\d')].columns)

#set the below variable to whatever columns you want to keep
columns_to_keep = columns_to_keep_top

MUST_HAVE_COLUMNS = ['Yield']
# print(columns_to_keep)

df = df.drop(columns_to_drop, axis=1) if should_drop else df
df = df.loc[:, np.concatenate((columns_to_keep, MUST_HAVE_COLUMNS))] if should_keep else df
df['YieldBucket'] = pd.Series(pd.qcut(df.Yield, q=3, labels=["high", "medium", "low"]))
print("The final dataframe has columns: ", df.columns)

# TRAIN AND TEST SPLIT# TRAIN AND TEST SPLIT# TRAIN AND TEST SPLIT
# TRAIN AND TEST SPLIT# TRAIN AND TEST SPLIT# TRAIN AND TEST SPLIT
# TRAIN AND TEST SPLIT# TRAIN AND TEST SPLIT# TRAIN AND TEST SPLIT
# TRAIN AND TEST SPLIT# TRAIN AND TEST SPLIT# TRAIN AND TEST SPLIT

X = df.drop(['Yield', 'YieldBucket'], axis=1)

print(X.columns)

y = df.Yield

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, train_size = 0.1, random_state = 42)

INPUT_COLS = X_train.columns
# TEST_COLS = y_train.columns

import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor

from sklearn.feature_selection import RFECV

classifiers = [
    svm.SVR(),
    MLPRegressor(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
#     linear_model.ARDRegression(),
#     linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()]

estimator = svm.SVR(kernel="linear")

selector = RFECV(estimator, step=1, cv=5, verbose=1, n_jobs = -1)
selector = selector.fit(X_train, y_train)
print(selector.support_)
# # array([ True,  True,  True,  True,  True,
# #         False, False, False, False, False], dtype=bool)
print(selector.ranking_)
# # array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])


#     print(np.sum(preds - y_test))
#     print(clf.predict(X_test),'\n')
#     print(y_test)
#     print('accuracy score:', accuracy_score(y_test, clf.predict(X_test)), '\n')