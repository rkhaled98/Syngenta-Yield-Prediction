import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# visulaize the important characteristics of the dataset
import matplotlib.pyplot as plt

# step 1: download the data
dataframe_all = pd.read_csv("Syngenta/Syngenta_2017/Experiment_dataset.csv")
num_rows = dataframe_all.shape[0]

# step 2: remove useless data
# count the number of missing elements (NaN) in each column
counter_nan = dataframe_all.isnull().sum()
counter_without_nan = counter_nan[counter_nan==0]
# remove the columns with missing elements
dataframe_all = dataframe_all[counter_without_nan.keys()]
# remove the first 7 columns which contain no discriminative information
# dataframe_all = dataframe_all.ix[:,7:]
# the list of columns (the last column is the class label)
columns = dataframe_all.columns
# print(columns)



dataframe_all = dataframe_all.drop(['Experiment', 'Location', 'Planting date', 'Check Yield', 'Yield difference', 'Latitude', 'Longitude', 'Variety', 'PI'], axis=1)


# step 3: get features (x) and scale the features
# get x and convert it to numpy array
x = dataframe_all.ix[:,:-1].values
standard_scaler = StandardScaler()
x_std = standard_scaler.fit_transform(x)

# step 4: get class labels y and then encode it into number 
# get class label data
# y = dataframe_all.ix[:,-1].values
y = dataframe_all.Yield

y = pd.qcut(y, q=3, labels=["high", "medium", "low"])

print(y)


print(y)
# encode the class label
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# step 5: split the data into training set and test set
test_percentage = 0.05
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_percentage, random_state = 0)

# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0, verbose=1)
x_test_2d = tsne.fit_transform(x_test)



# scatter plot the sample points among 5 classes
# markers=('s', 'd', 'o', '^', 'v')
markers=('s', 'd', 'o', '^')
# color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan'}
color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple'}
plt.figure()
for idx, cl in enumerate(np.unique(y_test)):
    try:
        plt.scatter(x=x_test_2d[y_test==cl,0], y=x_test_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
    except KeyError:
        print("Key Error")
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()