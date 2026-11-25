"""Model creation, training, evaluation and retraining (MobileNetV2 transfer learning)."""

import datetime
import os
import shutil

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras import layers, regularizers

from preprocessing import CLASS_NAMES, IMG_SIZE, NUM_CLASSES

BASE_NAME = "mobilenetv2_1.00_224"
MODEL_FILENAME = "bean_model.h5"


def build_model(num_classes=NUM_CLASSES, dropout=0.3, l2=1e-4, learning_rate=1e-3):
    """Frozen MobileNetV2 base + small dense head with dropout and L2."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)  # [0, 1] -> [-1, 1]
    x = base(x, training=False)                      # keep BatchNorm frozen
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(l2))(x)

    model = tf.keras.Model(inputs, outputs, name="bean_leaf_classifier")
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])


def set_trainable_layers(model, n_trainable):
    """Unfreeze the last n_trainable base layers (BatchNorm stays frozen)."""
    base = model.get_layer(BASE_NAME)
    base.trainable = n_trainable > 0
    first = len(base.layers) - n_trainable
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= first and not isinstance(layer, layers.BatchNormalization)


def default_callbacks(patience=4):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=max(1, patience // 2), min_lr=1e-7),
    ]


def train_model(model, train_ds, val_ds, epochs=15, patience=4, verbose=1):
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=default_callbacks(patience), verbose=verbose)


def fine_tune_model(model, train_ds, val_ds, n_trainable=40, learning_rate=1e-5,
                    epochs=10, patience=4, verbose=1):
    """Stage 2: unfreeze the top of the base and train with a low learning rate."""
    set_trainable_layers(model, n_trainable)
    compile_model(model, learning_rate)
    return train_model(model, train_ds, val_ds, epochs, patience, verbose)


def predict_dataset(model, ds):
    """Return (true labels, predicted probabilities) for a labelled dataset."""
    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_prob = model.predict(ds, verbose=0)  # graph mode: much less memory than a Python loop
    return y_true, y_prob


def evaluate_model(model, ds):
    """Accuracy, loss, macro precision/recall/F1 and confusion matrix."""
    y_true, y_prob = predict_dataset(model, ds)
    return compute_metrics(y_true, y_prob)


def compute_metrics(y_true, y_prob):
    y_pred = y_prob.argmax(axis=1)
    loss = -np.mean(np.log(y_prob[np.arange(len(y_true)), y_true] + 1e-7))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(loss),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
        "n_images": int(len(y_true)),
    }


def save_model(model, model_dir="../models"):
    """Save a timestamped copy in models/versions/ and make it models/bean_model.h5."""
    os.makedirs(os.path.join(model_dir, "versions"), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version_path = os.path.join(model_dir, "versions", f"bean_model_{stamp}.h5")
    latest_path = os.path.join(model_dir, MODEL_FILENAME)

    model.save(version_path, include_optimizer=False)
    shutil.copy2(version_path, latest_path + ".tmp")
    os.replace(latest_path + ".tmp", latest_path)  # atomic swap for the running API
    return latest_path, version_path


def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)


def retrain_model(model, train_ds, val_ds, epochs=5, learning_rate=1e-4,
                  n_trainable=0, patience=2, verbose=0, callbacks=None):
    """Continue training our saved model (a path or a loaded model). Returns (model, history).

    With n_trainable=0 the base stays frozen, so its features are computed once
    and only the head is trained. This is fast and needs little memory.
    """
    if isinstance(model, str):
        model = load_trained_model(model)

    if n_trainable > 0:
        set_trainable_layers(model, n_trainable)
        compile_model(model, learning_rate)
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                            callbacks=default_callbacks(patience) + (callbacks or []),
                            verbose=verbose)
        return model, history

    extractor, head = split_model(model)
    history = train_head(head, extract_features(extractor, train_ds),
                         extract_features(extractor, val_ds),
                         epochs, learning_rate, patience, verbose, callbacks)
    return model, history


def split_model(model):
    """Split at the pooling layer: frozen feature extractor + head (shares weights with model)."""
    pool = next(i for i, layer in enumerate(model.layers)
                if isinstance(layer, layers.GlobalAveragePooling2D))
    extractor = tf.keras.Model(model.input, model.layers[pool].output)
    head = tf.keras.Sequential(model.layers[pool + 1:])
    return extractor, head


def extract_features(extractor, ds):
    """Run the frozen base once. Returns (features, labels)."""
    features, labels = [], []
    for x, y in ds:
        features.append(extractor.predict_on_batch(x))
        labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)


def train_head(head, train, val, epochs=5, learning_rate=1e-4, patience=2,
               verbose=0, callbacks=None):
    """Train the head on precomputed (features, labels)."""
    compile_model(head, learning_rate)
    return head.fit(*train, validation_data=val, batch_size=16, epochs=epochs,
                    callbacks=default_callbacks(patience) + (callbacks or []),
                    verbose=verbose)
