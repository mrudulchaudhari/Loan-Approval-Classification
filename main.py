# 3 Oct 2025
# Importing EDA libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


#Loading dataset
df = pd.read_csv('loan_data.csv')


#BASIC EDA

#seeing top 10 entries of dataset
print("first 10 rows of dataset")
print(df.head(10))
print("-"*50)

#seeing info
print("information on data set column wise")
print(df.info())
print("-"*50)

#seeing count of null values column wise
print("Count of null values column wise")
print(df.isnull().sum())
print("-"*50)

# As there are no null values, we need not handle them

#seeing duplicate values
print("Count of duplicate values")
print(df.duplicated().sum())
print("-"*50)

# There are no duplicate values, so we need not handle them
