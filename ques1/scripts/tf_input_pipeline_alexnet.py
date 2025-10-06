"""
tf_input_pipeline_alexnet.py
----------------------------
TensorFlow/Keras input pipelines for the Tiny-ImageNet 100×500 subset built earlier.

Preprocessing to match AlexNet Section 2 (adapted to Tiny-ImageNet):
  - Resize shorter side to 256 (preserving aspect ratio)
  - 224×224 random crop (train) / center crop (val/test)
  - Random horizontal flip (train)
  - Per-channel mean/std normalization using dataset_stats.json (computed on train set)
  - Optional PCA lighting (AlexNet) as a custom augmentation (applied on [0,1] images)

Directory layout expected (created by prepare_tiny_imagenet_subset.py):
  <data_root>/
    ├─ train/<class>/*.JPEG  (300/class)
    ├─ val/<class>/*.JPEG    (100/class)
    ├─ test/<class>/*.JPEG   (100/class)
    ├─ classes.txt
    ├─ labels.json
    └─ dataset_stats.json

Usage:
  pip install tensorflow pillow numpy
  python tf_input_pipeline_alexnet.py --data-root "C:\path\to\data_subset_100x500" --batch-size 128 --use-pca-lighting

Notes:
  * This code reads file paths with Python and feeds them to tf.data so we can implement
    short-side resize to 256 followed by specific crops.
  * PCA lighting here uses the same default eigvals/eigvecs commonly used in AlexNet examples.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

# ---------- Utility to load stats ----------
def load_stats(stats_path: Path):
    with open(stats_path, "r") as f:
        stats = json.load(f)
    mean = tf.constant(stats["mean_rgb"], dtype=tf.float32)  # list[3]
    std  = tf.constant(stats["std_rgb"],  dtype=tf.float32)  # list[3]
    return mean, std

# ---------- File listing helpers ----------
def list_files_and_labels(split_dir: Path, class_to_id: dict) -> Tuple[List[str], List[int]]:
    filepaths = []
    labels = []
    for cname, cid in class_to_id.items():
        cdir = split_dir / cname
        if not cdir.exists():
            raise FileNotFoundError(f"Class folder not found: {cdir}")
        for p in cdir.iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                filepaths.append(str(p))
                labels.append(int(cid))
    return filepaths, labels

def load_class_id_map(data_root: Path) -> dict:
    # The order in classes.txt defines IDs; labels.json contains the same mapping.
    with open(data_root / "labels.json", "r") as f:
        return json.load(f)

# ---------- Image preprocessing ----------
def decode_image(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    # Decode and always convert to 3 channels RGB
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img

def resize_shorter_side_to(img: tf.Tensor, target: int) -> tf.Tensor:
    """Resize preserving aspect ratio so that the shorter side == target."""
    shape = tf.shape(img)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    shorter = tf.minimum(h, w)
    scale = tf.cast(target, tf.float32) / shorter
    new_h = tf.cast(tf.round(h * scale), tf.int32)
    new_w = tf.cast(tf.round(w * scale), tf.int32)
    img = tf.image.resize(img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR)
    return img

def random_crop_224(img: tf.Tensor) -> tf.Tensor:
    return tf.image.random_crop(img, size=(224, 224, 3))

def center_crop_224(img: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]
    top  = tf.cast(tf.maximum(0, (h - 224) // 2), tf.int32)
    left = tf.cast(tf.maximum(0, (w - 224) // 2), tf.int32)
    img = tf.image.crop_to_bounding_box(img, offset_height=top, offset_width=left, target_height=224, target_width=224)
    return img

def random_flip_lr(img: tf.Tensor) -> tf.Tensor:
    return tf.image.random_flip_left_right(img)

# ---------- PCA Lighting (AlexNet) ----------
class PCALighting(tf.keras.layers.Layer):
    """
    AlexNet 'PCA jitter' lighting augmentation.
    Applied to images in [0,1] range BEFORE normalization.
    eigval/eigvec defaults are widely used approximations for ImageNet.
    """
    def __init__(self, alphastd=0.1, eigval=None, eigvec=None, **kwargs):
        super().__init__(**kwargs)
        if eigval is None:
            eigval = [0.2175, 0.0188, 0.0045]
        if eigvec is None:
            eigvec = [
                [-0.5675,  0.7192,  0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948,  0.4203],
            ]
        self.alphastd = alphastd
        self.eigval = tf.constant(eigval, dtype=tf.float32)  # (3,)
        self.eigvec = tf.constant(eigvec, dtype=tf.float32)  # (3,3)

    def call(self, img, training=None):
        if not training or self.alphastd == 0.0:
            return img
        alpha = tf.random.normal((3,), mean=0.0, stddev=self.alphastd, dtype=tf.float32)  # (3,)
        rgb = tf.linalg.matvec(self.eigvec, self.eigval * alpha)  # (3,)
        # Broadcast add across H×W×C
        img = img + rgb
        return tf.clip_by_value(img, 0.0, 1.0)

def normalize(img: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
    # img is [0,1]; apply (img - mean)/std per channel
    return (img - mean) / std

# ---------- tf.data builders ----------
def make_dataset(filepaths, labels, batch_size, training, mean, std, use_pca=False, cache=False):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)

    def _load_and_preprocess(path, label):
        img = decode_image(path)
        img = resize_shorter_side_to(img, 256)
        if training:
            img = random_crop_224(img)
            img = random_flip_lr(img)
            if use_pca:
                img = PCALighting(alphastd=0.1)(img, training=True)
        else:
            img = center_crop_224(img)
        img = normalize(img, mean, std)
        return img, label

    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# ---------- High-level convenience ----------
def build_loaders(data_root: Path, batch_size: int, use_pca_lighting: bool):
    class_to_id = load_class_id_map(data_root)
    mean, std = load_stats(data_root / "dataset_stats.json")

    train_files, train_labels = list_files_and_labels(data_root / "train", class_to_id)
    val_files,   val_labels   = list_files_and_labels(data_root / "val",   class_to_id)
    test_files,  test_labels  = list_files_and_labels(data_root / "test",  class_to_id)

    train_ds = make_dataset(train_files, train_labels, batch_size, training=True, mean=mean, std=std,
                            use_pca=use_pca_lighting, cache=False)
    val_ds   = make_dataset(val_files, val_labels, batch_size, training=False, mean=mean, std=std, cache=False)
    test_ds  = make_dataset(test_files, test_labels, batch_size, training=False, mean=mean, std=std, cache=False)
    return train_ds, val_ds, test_ds

# ---------- Demo main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=False,
                        default=Path(r"C:\Users\asmae\Documents\WSU\Deep Learning\assignment2-DL\ques1\data_subset_100x500"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--use-pca-lighting", action="store_true")
    args = parser.parse_args()

    train_ds, val_ds, test_ds = build_loaders(args.data_root, args.batch_size, use_pca_lighting=args.use_pca_lighting)

    # Show one batch shape & few labels
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        for images, labels in ds.take(1):
            print(f"{name}: images={images.shape}, labels={labels.shape}, "
                  f"min={tf.reduce_min(images).numpy():.3f}, max={tf.reduce_max(images).numpy():.3f}")
            print("labels sample:", labels[:8].numpy().tolist())

if __name__ == "__main__":
    main()