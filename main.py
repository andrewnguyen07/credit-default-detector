import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.layers import Dense, Dropout

import uvicorn
from fastapi import FastAPI
from fastapi import File, UploadFile

from functions_api import clean_up
from functions_api import one_hot_encoding
from functions_api import data_scaling

app = FastAPI()

# load the model
model = tf.keras.models.load_model('/Users/andrewnguyen/gitfolder/default/app_api/default_detector')

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # load the csv file
    csv = file.filename
    data = pd.read_csv(csv)
    # pre-process the data
    id = data.iloc[:, 0]
    df = clean_up(data)
    df = one_hot_encoding(df)
    df = data_scaling(df)
    # # create and return prediction
    pred = model.predict(df)
    pred_label = [0 if x < 0.5 else 1 for x in pred]
    pred_df = zip(id, pred_label)
    result = {'result': pred_df}
    return result



