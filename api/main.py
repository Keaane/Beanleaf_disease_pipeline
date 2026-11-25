"""FastAPI app: prediction, data upload, retraining, evaluation and the web UI.

Run from the api/ folder:  uvicorn main:app --port 8000
"""

import hashlib
import json
import os
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import database  # noqa: E402
import retrain_job  # noqa: E402
from model import MODEL_FILENAME, evaluate_model  # noqa: E402
from prediction import load_model, predict_image  # noqa: E402
from preprocessing import CLASS_NAMES, class_counts, is_valid_image, list_images, make_dataset  # noqa: E402

MODEL_DIR = os.environ.get("BEAN_MODEL_DIR", os.path.join(ROOT, "models"))
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
TRAIN_DIR = os.path.join(ROOT, "data", "train")
TEST_DIR = os.path.join(ROOT, "data", "test")
UI_DIR = os.path.join(ROOT, "ui")

# Retraining starts by itself once this many new images are waiting.
RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", 30))
AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "1") == "1"

RETRAIN_CONFIG = {
    "model_path": MODEL_PATH,
    "model_dir": MODEL_DIR,
    "train_dir": TRAIN_DIR,
    "test_dir": TEST_DIR,
    "min_new_images": int(os.environ.get("RETRAIN_MIN_NEW_IMAGES", 3)),
    "replay_per_class": int(os.environ.get("RETRAIN_REPLAY_PER_CLASS", 30)),
    "batch_size": int(os.environ.get("RETRAIN_BATCH_SIZE", 16)),
    "epochs": int(os.environ.get("RETRAIN_EPOCHS", 5)),
    "fine_tune_layers": int(os.environ.get("RETRAIN_FINE_TUNE_LAYERS", 0)),
    "learning_rate": float(os.environ.get("RETRAIN_LR", 1e-4)),
}

START_TIME = time.time()
REPLICA = socket.gethostname()
_evaluation = {"model_time": None, "result": None}
_evaluation_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    database.init_db()
    load_model(MODEL_PATH)  # load once so the first prediction is fast
    yield


app = FastAPI(title="Bean Leaf Disease Classifier", version="2.0", lifespan=lifespan)
app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")


def model_info():
    info = {
        "file": MODEL_FILENAME,
        "size_mb": round(os.path.getsize(MODEL_PATH) / 1e6, 1),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S",
                                      time.localtime(os.path.getmtime(MODEL_PATH))),
        "training_test_metrics": None,
    }
    metrics_file = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            info["training_test_metrics"] = json.load(f)["test"]
    return info


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(UI_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "replica": REPLICA}


@app.get("/status")
def status():
    pending = database.count_pending_uploads()
    latest = database.latest_job()
    if latest:
        latest.pop("log")
    return {
        "status": "up",
        "replica": REPLICA,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(START_TIME)),
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model": model_info(),
        "predictions": database.prediction_stats(),
        "uploads_pending": sum(pending.values()),
        "uploads_pending_by_class": pending,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "auto_retrain": AUTO_RETRAIN,
        "latest_retrain": latest,
    }


# The endpoints below are plain `def`, so FastAPI runs them in a thread pool
# and a slow TensorFlow call never blocks other requests.

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    if not is_valid_image(image_bytes):
        raise HTTPException(400, "The file is not a readable image (use JPG or PNG).")

    start = time.perf_counter()
    result = predict_image(image_bytes=image_bytes, model_path=MODEL_PATH)
    latency_ms = (time.perf_counter() - start) * 1000

    database.log_prediction(result["predicted_class"], result["confidence"], latency_ms)
    return {**result, "latency_ms": round(latency_ms, 1), "replica": REPLICA}


@app.post("/upload-data/{class_name}")
def upload_data(class_name: str, files: list[UploadFile] = File(...)):
    """Save labelled images to the database for the next retraining."""
    if class_name not in CLASS_NAMES:
        raise HTTPException(400, f"class_name must be one of {CLASS_NAMES}")

    saved, duplicates, rejected = [], [], []
    for file in files:
        name = os.path.basename(file.filename or "image")
        image_bytes = file.file.read()
        if not is_valid_image(image_bytes):
            rejected.append(name)
        elif database.add_upload(class_name, name, hashlib.sha256(image_bytes).hexdigest(),
                                 image_bytes):
            saved.append(name)
        else:
            duplicates.append(name)

    pending = sum(database.count_pending_uploads().values())
    auto_job = None
    if AUTO_RETRAIN and pending >= RETRAIN_THRESHOLD:
        try:
            auto_job = retrain_job.start("auto", RETRAIN_CONFIG)
        except RuntimeError:
            pass  # a job is already running

    return {
        "class_name": class_name,
        "saved": len(saved),
        "duplicates": duplicates,
        "rejected": rejected,
        "pending_for_retrain": pending,
        "auto_retrain_started": auto_job,
    }


@app.get("/uploads")
def uploads(limit: int = 50):
    return database.recent_uploads(limit)


@app.post("/retrain", status_code=202)
def retrain():
    """Start retraining in the background. Follow it with GET /retrain/status."""
    try:
        job_id = retrain_job.start("manual", RETRAIN_CONFIG)
    except RuntimeError as error:
        raise HTTPException(409, str(error))
    return {"job_id": job_id, "status": "started"}


@app.get("/retrain/status")
def retrain_status():
    return database.latest_job() or {"status": "none"}


@app.get("/retrain/history")
def retrain_history():
    return database.job_history()


@app.post("/evaluate")
def evaluate():
    """Score the live model on the 128 test images (cached until the model changes)."""
    model_time = os.path.getmtime(MODEL_PATH)
    with _evaluation_lock:
        if _evaluation["model_time"] != model_time:
            start = time.perf_counter()
            test_ds = make_dataset(*list_images(TEST_DIR), batch_size=8)
            result = evaluate_model(load_model(MODEL_PATH), test_ds)
            result["seconds"] = round(time.perf_counter() - start, 1)
            result["model_last_updated"] = model_info()["last_updated"]
            result["class_names"] = CLASS_NAMES
            _evaluation.update(model_time=model_time, result=result)
        return _evaluation["result"]


@app.get("/visualizations/class-distribution")
def class_distribution():
    uploaded = database.upload_counts_by_class()
    return {
        "original": class_counts(TRAIN_DIR),
        "uploaded": {name: uploaded.get(name, 0) for name in CLASS_NAMES},
        "test": class_counts(TEST_DIR),
    }


@app.get("/visualizations/features")
def features():
    with open(os.path.join(UI_DIR, "data", "feature_summary.json")) as f:
        return json.load(f)
