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
print(df.head(10))

#seeing info
print(df.info())

#seeing count of null values column wise
print(df.isnull().sum())

# As there are no null values, we need not handle them

#seeing duplicate values
print(df.duplicated().sum())

# There are no duplicate values, so we need not handle them
