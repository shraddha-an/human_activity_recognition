# Human Activity Recognition: Feature Selection & Clustering

# Importing the libraries
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sb

# ============================  Building the DataFrame  =============================

# Extracting the column names from the features.txt file
c = []
with open('features.txt', 'r') as file:
    for line in file:
        c.append(line)

# Cleaning up column names to only include words
import re

cols = []
for x in c:
    word = re.sub('[^a-zA-Z]', '', x)
    cols.append(word)

# Extracting activity lables from the activity.txt file
a = []
with open('activity_labels.txt', 'r') as file:
    for line in file:
        a.append(line)

acts = []
for s in a:
    word = re.sub('[^a-zA-Z]', '', s)
    acts.append(word)

# Importing the dataset
dataset = pd.read_csv('X_train.csv', header = None)
dataset.columns = cols
X_train = dataset.iloc[:,:].values

ds = pd.read_csv('X_test.csv', header = None)
ds.columns = cols
X_test = ds.iloc[:,:].values

y_train = pd.read_csv('y_train.csv', header = None)
y_train = y_train.iloc[:,:].values

y_test = pd.read_csv('y_test.csv', header = None)
y_test = y_test.iloc[:,:].values


# ====================================  RFE & Classification  =============================

# Feature Selection
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
rfe = RFE(estimator = SVC(kernel = 'linear', random_state = 0), n_features_to_select = 100)

# Fitting RFE on the training model
rfe.fit(X_train, y_train)

# Saving Columns that were retained
retained_columns = rfe.support_

rc = []
for i in range(len(retained_columns)):
    if retained_columns[i]:
        rc.append(ds.columns[i])

# Transform the data
X = rfe.transform(X_train)
X_t = rfe.transform(X_test)

# Fitting SVC on the RFE data
model = SVC(kernel = 'linear', random_state = 0)
model.fit(X, y_train)

# Predictions
y_pred = model.predict(X_t).reshape(-1, 1)

# Metrics
from sklearn.metrics import classification_report
classification_report = classification_report(y_test, y_pred, target_names = acts)

# ====================================  Visualizations  =============================

# 1) Plotting actual & predicted values
plt.figure(figsize = (10, 12))
sb.set_style('darkgrid')
plt.hist([y_pred.flatten(), y_test.flatten()], label = ['Predictions','Actual Values'],
         density = True, color = ['#F06292','#3F51B5'], edgecolor = 'black')

plt.title('Predictions vs Actual Values')
plt.legend(loc = 'upper center', shadow = True, fontsize = 14)

plt.rc('axes', titlesize = 40)

plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)

#plt.savefig('actual_pred.png')
plt.show()
