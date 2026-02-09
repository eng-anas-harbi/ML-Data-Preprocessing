# This is a standard workflow for preparing data for machine learning models
# Suitable as an educational example
# Some steps might not be required for certain models (e.g., tree-based models do not need scaling)

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the dataset
# Split into:
# x: independent variables (features)
# y: dependent variable (target)
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Create an Imputer to handle missing values
# missing_values: the values to replace (NaN)
# strategy: method of replacement, here using the mean of numerical columns
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


# Fit the imputer only on numerical columns (Age, Salary)
# Do not include categorical columns
imputer.fit(x[:, 1:3])


# Apply the imputer to replace missing values in the same columns
x[:, 1:3] = imputer.transform(x[:, 1:3])


# ColumnTransformer is used here to encode only the categorical column (Country)
# OneHotEncoder:
# - removes the original column
# - creates a new column for each category
# - sets 1 if the category matches, 0 otherwise
# remainder='passthrough' keeps the other columns unchanged
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)


# Use fit_transform here because we are preprocessing the data
# The output is converted explicitly to a NumPy array
x = np.array(ct.fit_transform(x))


# The target variable y is binary (Yes / No)
# LabelEncoder is suitable here because:
# - it converts categorical values to 0 and 1
# - safe for binary classification
le = LabelEncoder()
y = le.fit_transform(y)


# Split data into training and testing sets
# Order matters: x first, then y
# test_size = 0.2 means 20% of data for testing
# random_state ensures reproducibility
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1
)


# StandardScaler is used to standardize numerical features
# This prevents columns with large values from dominating the model
sc = StandardScaler()


# Fit only on training data to calculate mean and standard deviation
# Columns from index 3 onward are numerical after OneHotEncoding
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])


# Transform test data using the same parameters
# Prevents data leakage
x_test[:, 3:] = sc.transform(x_test[:, 3:])

# Uncomment to print scaled data for verification
print("X_train after scaling:\n", x_train, "\n###########################\n")
print("X_test after scaling:\n", x_test, "\n###########################\n")
