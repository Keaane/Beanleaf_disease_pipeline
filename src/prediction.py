"""Predict the class of a single leaf image."""

import os
import threading

import numpy as np
import tensorflow as tf

from model import split_model
from preprocessing import CLASS_NAMES, preprocess_image_from_path, preprocess_uploaded_bytes

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "bean_model.h5")

_cache = {"key": None, "parts": None}
_cache_lock = threading.Lock()
model_lock = threading.Lock()  # one prediction (or retrain step) at a time keeps memory flat


def load_parts(model_path=DEFAULT_MODEL_PATH):
    """Return (model, extractor, head), loaded once and reloaded if the file changes.

    Predictions, evaluation and retraining all use these same objects, so the
    API only ever holds one copy of the network in memory.
    """
    key = (os.path.abspath(model_path), os.path.getmtime(model_path))
    with _cache_lock:
        if _cache["key"] != key:
            model = tf.keras.models.load_model(model_path, compile=False)
            _cache["parts"] = (model, *split_model(model))
            _cache["key"] = key
        return _cache["parts"]


def load_model(model_path=DEFAULT_MODEL_PATH):
    return load_parts(model_path)[0]


def mark_saved(model_path=DEFAULT_MODEL_PATH):
    """The cached model was retrained in memory and saved; keep using it."""
    with _cache_lock:
        _cache["key"] = (os.path.abspath(model_path), os.path.getmtime(model_path))


def predict_image(image_path=None, image_bytes=None, model_path=DEFAULT_MODEL_PATH):
    """Return the predicted class, its confidence and all class probabilities."""
    if image_bytes is not None:
        batch = preprocess_uploaded_bytes(image_bytes)
    else:
        batch = preprocess_image_from_path(image_path)

    _, extractor, head = load_parts(model_path)
    with model_lock:
        probs = head.predict_on_batch(extractor.predict_on_batch(batch))[0]
    best = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[best],
        "confidence": round(float(probs[best]), 4),
        "class_probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
    }
