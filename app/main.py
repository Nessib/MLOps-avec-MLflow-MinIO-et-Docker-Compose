# app/main.py
import os

# Variables d'environnement pour MinIO
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"  # URL de ton MinIO
os.environ["AWS_ACCESS_KEY_ID"] = "mlflow"  # Utilisateur MinIO
os.environ["AWS_SECRET_ACCESS_KEY"] = "mlflow123"  # Mot de passe MinIO

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn

app = FastAPI()
model = mlflow.sklearn.load_model("mlruns/0/<id_run>/artifacts/model")  # à adapter

class Input(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.post("/predict")
def predict(data: Input):
    input_data = [[data.feature1, data.feature2, data.feature3]]
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
