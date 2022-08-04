import pandas as pd

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('/Users/crystalli/Documents/Senior/AmazonPTA/DataFile.csv')
# Print the first five rows
# NaN means missing data

print(df.head())
print('The shape of the dataset is:', df.shape)
print(df.info())
print(df.describe())

import matplotlib.pyplot as plt

df['target_label'].value_counts().plot.bar()
plt.show()

import numpy as np
np.set_printoptions(threshold=np.inf) # use this for datasets with more columns, to print all columns

# This prints the column labels of the dataframe
print('All dataset columns:')
print(df.columns.values)

# This prints the column labels of the features identified as numerical
print('Numerical columns:')
print(df.select_dtypes(include=np.number).columns.values)

# This prints the column labels of the features identified as numerical
print('Categorical columns:')
print(df.select_dtypes(include='object').columns.values)

# Grab model features/inputs and target/output
numerical_features = ["ASIN_STATIC_ITEM_PACKAGE_WEIGHT",
                      "ASIN_STATIC_LIST_PRICE"]

model_features = numerical_features
model_target = 'target_label'

# Data Cleansing: Cleaning numerical features
for i in range(0,len(numerical_features)):
    print(df[numerical_features[i]].value_counts(bins=10, sort=False))

# Remove Outliers
# print(df[df[numerical_features[1]] > 3000000])
dropIndexes = df[df[numerical_features[1]] > 3000000].index
df.drop(dropIndexes , inplace=True)
df[numerical_features[1]].value_counts(bins=10, sort=False)

# Check Missing Value
print(df[numerical_features].isna().sum())

# Train Dataset
from sklearn.model_selection import train_test_split

training_data, test_data = train_test_split(df, test_size=0.1, shuffle=True, random_state=23)
train_data, val_data = train_test_split(training_data, test_size=0.15, shuffle=True, random_state=23)

# Print the shapes of the Train - Validation - Test Datasets
print('Train - Validation - Test Datasets shapes: ', train_data.shape, val_data.shape, test_data.shape)

# Data Processing with Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

### PIPELINE ###
################

# Pipeline desired data transformers, along with an estimator at the end
# For each step specify: a name, the actual transformer/estimator with its parameters
classifier = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler()),
    ('estimator', KNeighborsClassifier(n_neighbors = 3))
])

# Visualize the pipeline
# This will come in handy especially when building more complex pipelines, stringing together multiple preprocessing steps
from sklearn import set_config
set_config(display='diagram')
print(classifier)

# Train and Tune a Classifier
# Get train data to train the classifier
X_train = train_data[model_features]
y_train = train_data[model_target]

# Fit the classifier to the train data
# Train data going through the Pipeline it's imputed (with means from the train data),
#   scaled (with the min/max from the train data),
#   and finally used to fit the model
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Use the fitted model to make predictions on the train dataset
# Train data going through the Pipeline it's imputed (with means from the train data),
#   scaled (with the min/max from the train data),
#   and finally used to make predictions
train_predictions = classifier.predict(X_train)

print('Model performance on the train set:')
print(confusion_matrix(y_train, train_predictions))
print(classification_report(y_train, train_predictions))
print("Train accuracy:", accuracy_score(y_train, train_predictions))

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Get validation data to validate the classifier
X_val = val_data[model_features]
y_val = val_data[model_target]

# Use the fitted model to make predictions on the validation dataset
# Validation data going through the Pipeline it's imputed (with means from the train data),
#   scaled (with the min/max from the train data),
#   and finally used to make predictions
val_predictions = classifier.predict(X_val)

print('Model performance on the validation set:')
print(confusion_matrix(y_val, val_predictions))
print(classification_report(y_val, val_predictions))
print("Validation accuracy:", accuracy_score(y_val, val_predictions))

# Model Tunning
# Try different values of K and select the one producing the highest metric on the validation set
# (later, we will see how do to this more efficiently with library hyperparameter tuning functions)

K_values = [1, 2, 3, 4, 5, 6]

K_best = 0.0
val_score_best = 0.0
for K in K_values:
    classifier = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', MinMaxScaler()),
        ('estimator', KNeighborsClassifier(n_neighbors=K))
    ])
    classifier.fit(X_train, y_train)
    val_predictions = classifier.predict(X_val)
    val_acc = accuracy_score(y_val, val_predictions)
    print("K=%d, Validation accuracy: %f" % (K, val_acc))
    if val_acc > val_score_best:
        K_best = K
        val_score_best = val_acc

print("K_best=%d, Best Validation accuracy: %f" % (K_best, val_score_best))

# Finally, train the best model on the whole training dataset again, before testing on the test dataset

# Get the best classifier
classifier = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', MinMaxScaler()),
        ('estimator', KNeighborsClassifier(n_neighbors = K_best))
            ])

# Get training data to train the classifier once more
X_training = training_data[model_features]
y_training = training_data[model_target]

# Train the best classifier once more on all training dataset
classifier.fit(X_training, y_training)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Use the fitted model to make predictions on the train dataset
train_predictions = classifier.predict(X_train)

print('Model performance on the train set:')
print(confusion_matrix(y_train, train_predictions))
print(classification_report(y_train, train_predictions))
print("Train accuracy:", accuracy_score(y_train, train_predictions))

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Get test data to test the classifier
X_test = test_data[model_features]
y_test = test_data[model_target]

# Use the fitted model to make predictions on the test dataset
# Test data going through the Pipeline it's imputed (with means from the train data),
#   scaled (with the min/max from the train data),
#   and finally used to make predictions
test_predictions = classifier.predict(X_test)

print('Model performance on the test set:')
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print("Test accuracy:", accuracy_score(y_test, test_predictions))