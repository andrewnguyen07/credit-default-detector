import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf

# clean up the feature variables
def clean_up(df):
    df = df.loc[(df['x3'] != 0) & (df['x4'] != 0)]
    df = df.loc[(df['x3'] != 4) & (df['x3'] != 5) & (df['x3'] != 6)]
    df.x2 = df.x2.replace(2, 0)
    df.x3 = df.x3.replace(3, 0)
    df.x4 = df.x4.replace(3, 0)
    df = df.reset_index(drop=True)
    return df

# one-hot encode categorical variables
from keras.utils import np_utils

def one_hot_encoding(df):
    # one-hot encode 
    x2 = pd.DataFrame(np_utils.to_categorical(df.iloc[:, 2]))
    x3 = pd.DataFrame(np_utils.to_categorical(df.iloc[:, 3]))
    x4 = pd.DataFrame(np_utils.to_categorical(df.iloc[:, 4]))
    # drop the original columns
    df = df.drop(columns=['x2', 'x3', 'x4'])
    # concatenate all the encoded variables into the df
    df = pd.concat([df, x2, x3, x4], axis=1)
    return df

# scale the data
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

def data_scaling(df):
    # drop the id 
    df = df.drop(columns=['id'])
    # scale the dataset
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    # concatenate the idf & target variable into the dataframe
    return df

