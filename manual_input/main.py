import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.layers import Dense, Dropout
from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI

app = FastAPI()

# load the model
model = tf.keras.models.load_model('/Users/andrewnguyen/gitfolder/default/app/detector')

class Data(BaseModel):
    x1: int
    x2: int
    x3: int
    x4: int
    x5: int
    x6: int
    x7: int
    x8: int
    x9: int
    x10: int
    x11: int
    x12: int
    x13: int
    x14: int
    x15: int
    x16: int
    x17: int
    x18: int
    x19: int
    x20: int
    x21: int
    x22: int
    x23: int

@app.post('/predict')
async def predict(data: Data):
    df = pd.DataFrame(data.dict(), index=[0])
    pred = round(float(model.predict(df)),0)
    if pred == 1:
        result = 'Likely to default loan'
    else: 
        result = 'Not likely to default loan'
    return {'result': result}



