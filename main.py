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

#REMOVING OUTLIER USING IQR METHOD
# def remove_outlier(dataframe:pd.DataFrame, column:str) -> pd.DataFrame:
#     """docstring for remove_outlier"""
#     q1 = dataframe[column].quantile(0.25)
#     q3 = dataframe[column].quantile(0.75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
#     dataframe = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
#
#     return dataframe

#didnt use it as it was too strict

def cap_outlier(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Cap outliers in a column using the IQR method.
    Instead of removing rows, values beyond the bounds
    are replaced with the boundary values.
    """
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Cap values
    dataframe[column] = dataframe[column].clip(lower=lower_bound, upper=upper_bound)

    return dataframe


print(df.columns)
"""Index(['person_age', 'person_gender', 'person_education', 'person_income',
       'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
       'credit_score', 'previous_loan_defaults_on_file', 'loan_status'],
      dtype='object')
"""

# Columns to apply outlier detection / capping
num_cols_with_outliers = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score"
]

for i in num_cols_with_outliers:
    df = cap_outlier(df, i)

print(df)
