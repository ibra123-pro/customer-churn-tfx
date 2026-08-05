import os
import glob
from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()

# Fungsi helper untuk mencari path model secara dinamis
def get_latest_model_path():
    # Mencari pola folder di dalam output/serving_model/
    # Contoh: output/serving_model/*/saved_model.pb atau ./serving_model/*/saved_model.pb
    search_paths = [
        "./output/serving_model/*/saved_model.pb",
        "output/serving_model/*/saved_model.pb",
        "./serving_model/*/saved_model.pb"
    ]
    
    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            # Ambil folder yang paling baru/sesuai
            latest_model = max(matches, key=os.path.getmtime)
            # Kembalikan direktori foldernya (bukan file .pb nya)
            return os.path.dirname(latest_model)
    return None

@app.get("/")
def read_root():
    return {"message": "Telco Customer Churn API is running!"}

# Endpoint revisi untuk mengecek file dan isi direktori model secara otomatis
@app.get("/check-files")
def check_files():
    # 1. Cek daftar file di direktori utama
    root_files = os.listdir(".")
    
    # 2. Cari path model otomatis
    model_path = get_latest_model_path()
    model_exists = model_path is not None and os.path.exists(model_path)
    
    # 3. Cek isi file di dalam folder model jika ketemu
    model_contents = os.listdir(model_path) if model_exists else []
    
    return {
        "root_directory_contents": root_files,
        "detected_model_path": model_path,
        "model_folder_exists": model_exists,
        "model_folder_contents": model_contents
    }