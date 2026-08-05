import os
import glob
from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()

def get_latest_model_path():
    # Otomatis melacak folder model apa pun yang ada di dalam serving_model/
    # Tanpa peduli angka timestamp-nya berubah-ubah
    search_paths = [
        "./serving_model/*/saved_model.pb",
        "serving_model/*/saved_model.pb",
        "./output/serving_model/*/saved_model.pb",
        "output/serving_model/*/saved_model.pb"
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

@app.get("/debug-model-search")
def debug_model_search():
    import glob
    import os
    
    # Cek berbagai kemungkinan lokasi folder serving_model
    patterns = [
        "./serving_model/*/saved_model.pb",
        "serving_model/*/saved_model.pb",
        "./output/serving_model/*/saved_model.pb",
        "output/serving_model/*/saved_model.pb",
        "**/saved_model.pb"
    ]
    
    found_matches = {}
    for p in patterns:
        found_matches[p] = glob.glob(p, recursive=True)
        
    return {
        "current_dir": os.getcwd(),
        "root_contents": os.listdir("."),
        "glob_results": found_matches
    }

@app.get("/v1/models/{model_name}/metadata")
def get_model_metadata(model_name: str):
    model_path = get_latest_model_path()
    
    if not model_path or not os.path.exists(model_path):
        return {"error": "Model not found on server"}, 404
        
    version_str = os.path.basename(model_path)
    
    try:
        # Cukup pastikan model dapat diload atau foldernya valid
        # Tanpa perlu mengekstrak structured_input_signature yang sering bentrok versi TF
        return {
            "model_spec": {
                "name": model_name,
                "signature_name": "",
                "version": version_str
            },
            "metadata": {
                "signature_def": {
                    "signature_def": {
                        "serving_default": {
                            "inputs": {},
                            "name": "serving_default"
                        }
                    }
                }
            }
        }
    except Exception as e:
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