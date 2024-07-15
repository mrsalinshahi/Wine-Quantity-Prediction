#Importing libraries and Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings
#warnings.filterwarnings('ignore')


# look at the first five rows of the dataset
df = pd.read_csv('winequality.csv')
print(df.head())


# explore the type of data present in each of the columns present in the dataset
df.info()

# explore the descriptive statistical measures of the dataset
df.describe().T


# check the number of null values in the dataset columns wise
df.isnull().sum()


# impute the missing values by means as the data present in the different columns are continuous values
for col in df.columns:
  if df[col].isnull().sum() > 0:
    df[col] = df[col].fillna(df[col].mean())
 
df.isnull().sum().sum()

# draw the histogram to visualise the distribution of the data
df.hist(bins=20, figsize=(10, 10))
plt.show()


# draw the count plot to visualise the number data for each quality of wine
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# we remove redundant features before to train our model.
plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()

df = df.drop('total sulfur dioxide', axis=1)


# prepare our data for training and splitting it into training and validation
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]


# object data type replace it with the 0 and 1
df.replace({'white': 1, 'red': 0}, inplace=True)


# split it into 80:20 ratio for model selection
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']
 
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)
 
xtrain.shape, xtest.shape

# Normalising the data 
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)


# train some state of the art machine learning model on it
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
 
for i in range(3):
    models[i].fit(xtrain, ytrain)
 
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(ytest, models[i].predict(xtest)))
    print()

# plot the confusion matrix as well for the validation data using the Logistic Regression model
cm = confusion_matrix(ytest, models[1].predict(xtest))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# print the classification report for the best performing model.
print(metrics.classification_report(ytest, models[1].predict(xtest)))

