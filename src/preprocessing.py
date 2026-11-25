"""Data loading and image preprocessing, shared by the notebook and the API."""

import os
import random

import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------- data acquisition ----------

def download_ibean(out_dir="../data"):
    """Download iBean from Hugging Face into data/train and data/test."""
    from datasets import load_dataset

    dataset = load_dataset("AI-Lab-Makerere/beans")
    splits = {"train": ["train", "validation"], "test": ["test"]}
    counts = {}
    for target, sources in splits.items():
        counts[target] = {name: 0 for name in CLASS_NAMES}
        for name in CLASS_NAMES:
            os.makedirs(os.path.join(out_dir, target, name), exist_ok=True)
        for source in sources:
            for example in dataset[source]:
                name = CLASS_NAMES[int(example["labels"])]
                n = counts[target][name]
                path = os.path.join(out_dir, target, name, f"{name}_{n:04d}.jpg")
                example["image"].convert("RGB").save(path, quality=95)
                counts[target][name] += 1
    return counts


def list_images(data_dir):
    """Return image paths and integer labels from data_dir/<class>/."""
    paths, labels = [], []
    for label, name in enumerate(CLASS_NAMES):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            continue
        for file in sorted(os.listdir(folder)):
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(folder, file))
                labels.append(label)
    return paths, labels


def class_counts(data_dir):
    _, labels = list_images(data_dir)
    return {name: labels.count(i) for i, name in enumerate(CLASS_NAMES)}


def stratified_split(paths, labels, val_fraction=0.15, seed=42):
    """Split into train and validation sets with the same class balance."""
    rng = random.Random(seed)
    train_p, train_l, val_p, val_l = [], [], [], []
    for label in sorted(set(labels)):
        items = [p for p, l in zip(paths, labels) if l == label]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_fraction))
        val_p += items[:n_val]
        val_l += [label] * n_val
        train_p += items[n_val:]
        train_l += [label] * (len(items) - n_val)
    return train_p, train_l, val_p, val_l


# ---------- preprocessing ----------

def decode_image(image_bytes):
    return tf.io.decode_image(image_bytes, channels=3, expand_animations=False)


def is_valid_image(image_bytes):
    try:
        image = decode_image(image_bytes)
        return image.shape[0] >= 32 and image.shape[1] >= 32
    except Exception:
        return False


def preprocess_image(image, label=None, augment=False):
    """Resize to 224x224 and scale to [0, 1]. Augment only for training."""
    image = tf.image.resize(tf.cast(image, tf.float32), IMG_SIZE) / 255.0

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        size = tf.cast(tf.random.uniform([], 0.8, 1.0) * IMG_SIZE[0], tf.int32)
        image = tf.image.random_crop(image, [size, size, 3])
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.image.random_brightness(image, 0.15)
        image = tf.image.random_contrast(image, 0.85, 1.15)
        image = tf.image.random_saturation(image, 0.85, 1.15)
        image = tf.clip_by_value(image, 0.0, 1.0)

    return image if label is None else (image, label)


def _batch(ds, batch_size, augment, shuffle, seed):
    if shuffle:
        ds = ds.shuffle(10_000, seed=seed)
    ds = ds.map(lambda img, lbl: preprocess_image(img, lbl, augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_dataset(paths, labels, batch_size=32, augment=False, shuffle=False, seed=42):
    """Batched dataset that reads image files from disk."""
    ds = tf.data.Dataset.from_tensor_slices((paths, tf.constant(labels, tf.int64)))
    ds = ds.map(lambda p, l: (decode_image(tf.io.read_file(p)), l))
    return _batch(ds, batch_size, augment, shuffle, seed)


def make_dataset_from_bytes(images, labels, batch_size=16, augment=False, shuffle=False, seed=42):
    """Batched dataset from raw image bytes (uploads stored in the database)."""
    ds = tf.data.Dataset.from_tensor_slices((tf.constant(images), tf.constant(labels, tf.int64)))
    ds = ds.map(lambda b, l: (decode_image(b), l))
    return _batch(ds, batch_size, augment, shuffle, seed)


def preprocess_uploaded_bytes(image_bytes):
    """One image as bytes -> batch of shape (1, 224, 224, 3)."""
    return tf.expand_dims(preprocess_image(decode_image(image_bytes)), 0)


def preprocess_image_from_path(path):
    with open(path, "rb") as f:
        return preprocess_uploaded_bytes(f.read())


# ---------- features for the visualisations ----------

def image_features(path, size=128):
    """Brightness, green fraction, lesion-coloured fraction and edge density."""
    image = tf.image.resize(decode_image(tf.io.read_file(path)), (size, size)).numpy()
    r, g, b = image[..., 0], image[..., 1], image[..., 2]

    green = (g > r * 1.05) & (g > b * 1.05)
    lesion = (r >= g * 0.95) & (r > b * 1.15) & (g > b * 0.9) & (r > 60)

    gray = image.mean(axis=-1)
    dx = np.abs(np.diff(gray, axis=1))[:-1, :]
    dy = np.abs(np.diff(gray, axis=0))[:, :-1]
    edges = np.sqrt(dx ** 2 + dy ** 2) > 40

    return {
        "brightness": float(image.mean()),
        "green_fraction": float(green.mean()),
        "lesion_fraction": float(lesion.mean()),
        "edge_density": float(edges.mean()),
    }
