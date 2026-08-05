import os
import glob
from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()

# Fungsi untuk mendeteksi path folder model terbaru secara otomatis
def get_latest_model_path():
    search_paths = [
        "./output/serving_model/*/saved_model.pb",
        "output/serving_model/*/saved_model.pb",
        "./serving_model/*/saved_model.pb"
    ]
    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            latest_model = max(matches, key=os.path.getmtime)
            return os.path.dirname(latest_model)
    return None

@app.get("/")
def read_root():
    return {"message": "Telco Customer Churn API is running!"}

# Endpoint metadata yang disamakan formatnya dengan standar TensorFlow Serving / mentor
@app.get("/v1/models/{model_name}/metadata")
def get_model_metadata(model_name: str):
    model_path = get_latest_model_path()
    
    if not model_path or not os.path.exists(model_path):
        return {"error": "Model not found on server"}, 404
        
    # Ambil versi (timestamp) dari nama folder terakhir
    version_str = os.path.basename(model_path)
    
    try:
        # Load SavedModel secara native menggunakan TensorFlow untuk membaca metadata aslinya
        saved_model_loaded = tf.saved_model.load(model_path)
        signatures = saved_model_loaded.signatures
        
        # Ekstrak signature_def ke dalam format dictionary agar mirip TensorFlow Serving
        sig_defs = {}
        for sig_key, sig_value in signatures.items():
            # Mengambil struktur input tensor dari signature model TFX
            inputs_meta = {}
            for input_name, tensor_spec in sig_value.structured_input_signature[1].items():
                inputs_meta[input_name] = {
                    "dtype": str(tensor_spec.dtype.name).upper(),
                    "tensor_shape": {
                        "dim": [{"size": str(dim)} for dim in tensor_spec.shape.as_list()] if tensor_spec.shape.as_list() else [{"size": "-1"}]
                    }
                }
            
            sig_defs[sig_key] = {
                "inputs": inputs_meta,
                "name": sig_key
            }

        return {
            "model_spec": {
                "name": model_name,
                "signature_name": "",
                "version": version_str
            },
            "metadata": {
                "signature_def": {
                    "signature_def": sig_defs
                }
            }
        }
    except Exception as e:
        # Fallback cadangan jika pembacaan signature internal gagal
        return {
            "model_spec": {
                "name": model_name,
                "signature_name": "",
                "version": version_str
            },
            "metadata": {
                "error": str(e)
            }
        }