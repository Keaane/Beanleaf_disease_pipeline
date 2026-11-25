"""Background retraining job.

Steps: load uploads from the database -> preprocess -> score the current model
-> continue training its head -> score the new model -> keep it only if F1 is not worse.
"""

import ctypes
import gc
import os
import random
import threading
import time
import traceback

import tensorflow as tf

import database
from model import compute_metrics, extract_features, save_model, train_head
from prediction import load_parts, mark_saved, model_lock
from preprocessing import (CLASS_NAMES, is_valid_image, list_images, make_dataset,
                           make_dataset_from_bytes, stratified_split)

_lock = threading.Lock()


def start(trigger, cfg):
    """Start a job in a background thread and return its id."""
    if database.running_job() or not _lock.acquire(blocking=False):
        raise RuntimeError("A retraining job is already running.")

    pending = sum(database.count_pending_uploads().values())
    if pending < cfg["min_new_images"]:
        _lock.release()
        raise RuntimeError(f"Upload at least {cfg['min_new_images']} new images first "
                           f"({pending} waiting).")

    job_id = time.strftime("%Y%m%d-%H%M%S")
    database.start_job(job_id, trigger)
    threading.Thread(target=_run, args=(job_id, cfg), daemon=True).start()
    return job_id


def _run(job_id, cfg):
    log_lines = []
    head = old_weights = None

    def log(message, stage):
        log_lines.append(time.strftime("%H:%M:%S ") + message)
        database.update_job(job_id, stage=stage, log="\n".join(log_lines))

    try:
        # 1. Load the new images from the database
        log("Loading uploaded images from the database", "loading")
        uploads = [u for u in database.pending_uploads() if is_valid_image(u[2])]
        new_images = [u[2] for u in uploads]
        new_labels = [CLASS_NAMES.index(u[1]) for u in uploads]
        log(f"  {len(new_images)} new images", "loading")

        # 2. Preprocess: add a sample of the original data so the model does not forget it
        log("Preprocessing (resize 224x224, scale to [0,1], augment)", "preprocessing")
        replay_images, replay_labels = _replay_sample(cfg["train_dir"], cfg["replay_per_class"])
        images = new_images + replay_images
        labels = new_labels + replay_labels
        train_idx, _, val_idx, _ = stratified_split(list(range(len(images))), labels, 0.2)
        train_ds = make_dataset_from_bytes([images[i] for i in train_idx],
                                           [labels[i] for i in train_idx],
                                           cfg["batch_size"], augment=True, shuffle=True)
        val_ds = make_dataset_from_bytes([images[i] for i in val_idx],
                                         [labels[i] for i in val_idx], cfg["batch_size"])
        test_ds = make_dataset(*list_images(cfg["test_dir"]), batch_size=cfg["batch_size"])
        log(f"  + {len(replay_images)} original images; "
            f"{len(train_idx)} train / {len(val_idx)} validation", "preprocessing")

        # 3. Run the frozen MobileNetV2 base once on every image
        log("Extracting MobileNetV2 features", "preprocessing")
        model, extractor, head = load_parts(cfg["model_path"])  # same objects as predictions
        train = extract_features(extractor, train_ds)
        val = extract_features(extractor, val_ds)
        x_test, y_test = extract_features(extractor, test_ds)

        # 4. Score the current model on the test set
        old = compute_metrics(y_test, head.predict(x_test, verbose=0))
        log(f"  current model: accuracy {old['accuracy']:.4f}, F1 {old['f1']:.4f}", "evaluating")

        # 5. Continue training the current model's head
        log(f"Retraining from the current model (up to {cfg['epochs']} epochs)", "training")
        old_weights = head.get_weights()
        with model_lock:  # predictions wait for the few seconds of training
            history = train_head(head, train, val, epochs=cfg["epochs"],
                                 learning_rate=cfg["learning_rate"],
                                 callbacks=[EpochLogger(log)])

            # 6. Score the new model and decide
            new = compute_metrics(y_test, head.predict(x_test, verbose=0))
            log(f"  retrained model: accuracy {new['accuracy']:.4f}, F1 {new['f1']:.4f}",
                "evaluating")

            promoted = new["f1"] >= old["f1"]
            version = None
            if promoted:
                version = os.path.basename(save_model(model, cfg["model_dir"])[1])
                mark_saved(cfg["model_path"])
                log(f"Promoted: the new model is live ({version})", "done")
            else:
                head.set_weights(old_weights)  # roll back
                log("Not promoted: the new model scored lower, the old one stays live", "done")

        database.mark_uploads_used([u[0] for u in uploads], job_id)
        database.update_job(
            job_id, status="completed", finished_at=time.time(),
            new_images=len(new_images), replay_images=len(replay_images),
            epochs_run=len(history.history["loss"]),
            old_accuracy=old["accuracy"], old_f1=old["f1"],
            new_accuracy=new["accuracy"], new_f1=new["f1"],
            promoted=int(promoted), model_version=version)
    except Exception as error:
        if old_weights is not None:
            head.set_weights(old_weights)  # never leave a half-trained model live
        traceback.print_exc()
        log(f"FAILED: {error}", "failed")
        database.update_job(job_id, status="failed", finished_at=time.time(), error=str(error))
    finally:
        free_memory()
        _lock.release()


def free_memory():
    """Return freed memory to the OS (matters on a 512 MB instance)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # not Linux


def _replay_sample(train_dir, per_class):
    paths, labels = list_images(train_dir)
    images, sample_labels = [], []
    for label in range(len(CLASS_NAMES)):
        class_paths = [p for p, l in zip(paths, labels) if l == label]
        for path in random.sample(class_paths, min(per_class, len(class_paths))):
            with open(path, "rb") as f:
                images.append(f.read())
            sample_labels.append(label)
    return images, sample_labels


class EpochLogger(tf.keras.callbacks.Callback):
    """Write each epoch's metrics to the job log."""

    def __init__(self, log):
        super().__init__()
        self.log = log

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.log(f"  epoch {epoch + 1}: acc {logs.get('accuracy', 0):.4f}, "
                 f"val_acc {logs.get('val_accuracy', 0):.4f}, "
                 f"val_loss {logs.get('val_loss', 0):.4f}", "training")
