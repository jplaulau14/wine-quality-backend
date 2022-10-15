from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
import joblib

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load('gb_model.pkl')
sc = joblib.load('scaler.pkl')

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/predict/")
async def input(sulphates: float, alcohol: float, volatile_acidity: float, total_sulfur_dioxide: float, density: float):
    wine = pd.DataFrame([[sulphates, alcohol, volatile_acidity, total_sulfur_dioxide, density]], columns=['sulphates', 'alcohol', 'volatile acidity', 'total sulfur dioxide', 'density'])
    # Convert to dataframe
    wine_new = sc.transform(wine)
    prediction = predict(wine_new)
    cat = str(prediction[0])
    print(cat)
    return {"prediction": cat}

def predict(wine):
    prediction = model.predict(wine)
    return prediction
