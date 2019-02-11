# THERE ARE SEPARATE NOTEBOOKS FOR VISUALIZATIONS, DATASET ANALYSIS, ETC. IN THE REPO.

import pandas as pd
import numpy as np

# READ THE CSV INTO DATAFRAME

df = pd.read_csv('Syngenta/Syngenta_2017/Experiment_dataset.csv')

# CURRENTLY NECESSARY IF: USING 174 ADDITIONAL VARIETY COLUMNS METHOD

# THIS IS A DIFFERENT APPROACH TO THE ABOVE FOUR CELLS, WHERE WE HAVE 174 ADDITIONAL FEATURE COLUMNS
# EACH WITH A 0 (IF IT IS NOT OF THAT VARIETY) OR A 1 (IF IT IS OF THAT VARIETY)

variety_dummies = pd.get_dummies(df.Variety)
df = pd.concat([df, variety_dummies], axis=1)

# GOAL OF THIS MODULE:
# Encode the planting date as a season
# NEW GOAL:
# GET DUMMIES FOR SEASONS

# remove the dates that are "."
df = df[~df['Planting date'].str.match("\.")]
plant_date = df['Planting date'].apply(lambda dt: pd.to_datetime(dt))
plant_months = plant_date.apply(lambda dt: dt.month)
season = plant_date.rename("Season")
season = pd.to_datetime(season)
season = season.apply(lambda dt: (dt.month%12 + 3)//3)
# df['Plant date'] = pd.to_datetime(df['Plant date'])
df = pd.concat([df, season], axis=1)

# plant_date = pd.to_datetime(df['Planting date'], infer_datetime_format=True)
# df = df['Planting date'].apply(lambda dt: (dt.month%12 + 3)//3)
# pd.get_dummies(df['Planting date'])


# ADD MONTH OF MAY AND JUNE ONE HOT ENCODING INTO THE DATAFRAME
pd.get_dummies(plant_months).sum()
june = pd.get_dummies(plant_months).loc[:,6]
june = june.rename("June")
may = pd.get_dummies(plant_months).loc[:,5]
may = may.rename("May")
df = pd.concat([df, may], axis=1)
df = pd.concat([df, june], axis=1)

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
                   'Longitude', 'Variety', 'PI', 'Planting date', 'Season']

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

# LET US ALSO MAKE SURE THERE ARE NO NAN IN THE DATA
print("We expect to be %s nan values and there actually are %s nan values\n" % (0, np.sum(df.isnull().sum())))
print(df.isnull().sum())
# AFTER COLUMNS, MAKE SURE NO SKETCHY ONES
for col in df.columns:
    print(col, type(df[col][0]))    

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

print("X_train shape:", X_train.shape, "\ny_train shape:", y_train.shape)

# This function will evaluate the errors based on RMSE (from the challenge spec)
# also will evaluate based on average error

from sklearn.metrics import mean_squared_error
def evaluate_errors(prediction, actual):
    print("RMSE Error: ", np.sqrt(mean_squared_error(prediction, actual)))
    avg_error_vector = np.absolute(((preds - y_test) / y_test) * 100)
#     print("Average Error: ", np.mean(avg_error_vector))
    print("Average Error details:\n", avg_error_vector.describe())
    return avg_error_vector

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators=20, max_depth=13, random_state=0, verbose=1)
regr.fit(X_train, y_train)
preds = regr.predict(X_test)

evaluate_errors(preds, y_test)

# GET OUTPUT OF FEATURE IMPORTANCE
def get_feature_importances(regr):
    feature_importances = regr.feature_importances_
    feature_importances = pd.Series(feature_importances)
    feature_importance_df = pd.DataFrame({'feature': X_train.columns,'feature_importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by=['feature_importance'])
    for index, row in feature_importance_df.iterrows():
        print(row['feature'], 'has importance: ', row['feature_importance'])
get_feature_importances(regr)

import random

NUM_VARIETIES = 174

def best_yield_variety(regr, test_set, random_sel = True, n_samples = 174, print_variety_preds = True):
    
    #create empty df
    dup_df = pd.DataFrame()
    
    #choose a rand sample of input test_set (for dev purposes, wouldn't be used in app)
    test_set_sample = test_set.sample(n= n_samples) if random_sel else test_set
    
    #progress of intensive, long for loop upcoming
    counter = 0
    
    #for loop will, for each row in test_set_sample, duplicate that row by NUM_VARIETIES
    for index, row in test_set_sample.iterrows():
        counter+=1
        print(counter)
        dup_df = dup_df.append([row] * NUM_VARIETIES, ignore_index = True)
        
    #extract the varieties columns
    duplicated_df_varieties = dup_df.loc[:, dup_df.columns.str.match('V\d\d\d\d\d\d')]
    #extract the names of the varieties
    varieties_array = duplicated_df_varieties.columns
    num_expanded_data_pts = duplicated_df_varieties.shape[0]
    #we must have a zeroed matrix of the same shape as the duplicated_df_varieties
    #so that we can input a 1 just once for each variety per 174 rows
    d = np.zeros((duplicated_df_varieties.shape[0], NUM_VARIETIES))
    #make d our dataframe, with columns equal to the varieties
    duplicated_df_varieties = pd.DataFrame(d, columns=varieties_array)
    #for loop will place a 1 just once for each variety per 174 rows (one hot rep)
    for i in range(duplicated_df_varieties.shape[0]):
        var_index = i % 174
        duplicated_df_varieties.loc[i, varieties_array[var_index]] = 1
    #remove the varieties (will be added back with the new values)
    dup_df = dup_df.drop(varieties_array, axis = 1)
    #add the new values (one hot representations) from above for loop
    dup_df = pd.concat([dup_df, duplicated_df_varieties], axis=1)
    #do prediction on the entire dataframe
    preds_per_variety = regr.predict(dup_df)
    #*******make it into a dataframe where each row will give the performance of each variety
    #with the same environmental conditions*******
    preds_df = pd.DataFrame(preds_per_variety.reshape((int(num_expanded_data_pts/NUM_VARIETIES), NUM_VARIETIES)),
                            columns=varieties_array)
    
    #environmental conditions (everything except the variety data)
    envcond = test_set_sample.drop(varieties_array, axis=1)
    
    #a simple print out for best variety given the environmental conditions
    hr_preds = []
    if print_variety_preds:
        envcond_cols = envcond.columns
        counter = -1
        for idx, row in envcond.iterrows():
            counter+=1
            out = "For environmental conditions:\n%s\nthe best variety is:%s" % (row, preds_df.idxmax(axis=1)[counter])
            hr_preds.append(out)
            print(out)
    
    return preds_df, envcond, hr_preds
    
        
# preds_df, envcond, hr_preds = best_yield_variety(regr, X_test, n_samples = 174)

# THIS WILL ONLY WORK WITH THE BUCKET METHOD

# from sklearn.ensemble import RandomForestClassifier
# regr = RandomForestClassifier(n_estimators=10, max_depth=20, random_state=0, verbose=1)
# regr.fit(X_train, y_train)
# preds = regr.predict(X_test)

# from sklearn.metrics import accuracy_score

# print(accuracy_score(y_test, preds))

# import numpy as np
# from sklearn import linear_model
# from sklearn import svm
# from sklearn.metrics import accuracy_score
# from sklearn.neural_network import MLPRegressor

# from sklearn.feature_selection import RFECV

# classifiers = [
#     svm.SVR(),
#     MLPRegressor(solver='lbfgs', alpha=1e-5,
#                      hidden_layer_sizes=(5, 2), random_state=1),
#     linear_model.SGDRegressor(),
#     linear_model.BayesianRidge(),
#     linear_model.LassoLars(),
# #     linear_model.ARDRegression(),
# #     linear_model.ARDRegression(),
#     linear_model.PassiveAggressiveRegressor(),
#     linear_model.TheilSenRegressor(),
#     linear_model.LinearRegression()]

# # estimator = svm.SVR(kernel="linear")

# # selector = RFECV(estimator, step=1, cv=5, verbose=1)
# # selector = selector.fit(X_train, y_train)
# # selector.support_ 
# # # array([ True,  True,  True,  True,  True,
# # #         False, False, False, False, False], dtype=bool)
# # selector.ranking_
# # # array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])


# #     print(np.sum(preds - y_test))
# #     print(clf.predict(X_test),'\n')
# #     print(y_test)
# #     print('accuracy score:', accuracy_score(y_test, clf.predict(X_test)), '\n')

# for item in classifiers:
#     print(item)
#     clf = item
#     clf.fit(X_train, y_train)
#     preds = clf.predict(X_test)
#     errors = evaluate_errors(preds, y_test)
#     try:
#         get_feature_importances(clf)
#     except:
#         print("NO FEATURE IMPORTANCE METRIC")