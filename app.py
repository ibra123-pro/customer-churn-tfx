from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import os

app = FastAPI()

# Menentukan path ke saved_model.pb hasil TFX pipeline
# Sesuaikan direktori penyimpanannya jika berbeda
MODEL_DIR = "./serving_model" 

model = None
if os.path.exists(MODEL_DIR):
    # Memuat model Keras/TensorFlow dari direktori TFX
    model = tf.keras.models.load_model(MODEL_DIR)

@app.get("/")
def read_root():
    return {"message": "Telco Customer Churn API is running!"}

@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model not found or loaded yet."}

    # Tambahkan logika pengolahan data dan prediksi di sini sesuai format input modelmu
    return {"status": "success", "message": "Endpoint siap menerima data prediksi"}