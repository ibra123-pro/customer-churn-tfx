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

@app.get("/check-files")
def check_files():
    # 1. Cek apakah folder serving_model ada
    folder_exists = os.path.exists(MODEL_DIR)
    
    # 2. Cek daftar file di dalam folder proyek utama
    root_files = os.listdir(".")
    
    # 3. Cek daftar file di dalam folder serving_model (jika ada)
    model_files = os.listdir(MODEL_DIR) if folder_exists else []
    
    return {
        "folder_serving_model_exists": folder_exists,
        "root_directory_contents": root_files,
        "serving_model_contents": model_files
    }