import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_hour_feature(df):
    df=df.copy()
    # the time column is in seconds, dividing by 3600 to get hrs and then 
    # mod 24 to get the hr of the day
    df["Hour"]=(df["Time"]//3600)%24
    df=df.drop("Time",axis=1)
    return df

def scale_amounts(X_train, X_test):
    scaler=StandardScaler()
    # making copies so we dont end up modifying the original df
    X_train=X_train.copy()
    X_test=X_test.copy()
    X_train["Amount"]=scaler.fit_transform(X_train[["Amount"]])
    # transforming the test set using the same scaler fitted on the training set
    X_test["Amount"]=scaler.transform(X_test[["Amount"]])
    return X_train, X_test
