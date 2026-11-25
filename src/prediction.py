"""Predict the class of a single leaf image."""

import os
import threading

import numpy as np
import tensorflow as tf

from preprocessing import CLASS_NAMES, preprocess_image_from_path, preprocess_uploaded_bytes

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "bean_model.h5")

_cache = {"key": None, "model": None}
_lock = threading.Lock()
_predict_lock = threading.Lock()  # one prediction at a time keeps memory flat under load


def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load the model once; reload it when the file changes after a retrain."""
    key = (os.path.abspath(model_path), os.path.getmtime(model_path))
    with _lock:
        if _cache["key"] != key:
            _cache["model"] = tf.keras.models.load_model(model_path, compile=False)
            _cache["key"] = key
        return _cache["model"]


def clear_model_cache():
    with _lock:
        _cache["key"] = _cache["model"] = None


def predict_image(image_path=None, image_bytes=None, model_path=DEFAULT_MODEL_PATH):
    """Return the predicted class, its confidence and all class probabilities."""
    if image_bytes is not None:
        batch = preprocess_uploaded_bytes(image_bytes)
    else:
        batch = preprocess_image_from_path(image_path)

    model = load_model(model_path)
    with _predict_lock:
        probs = model(batch, training=False).numpy()[0]
    best = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[best],
        "confidence": round(float(probs[best]), 4),
        "class_probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
    }
