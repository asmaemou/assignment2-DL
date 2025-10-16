"""
To summarize, this code prepares a subset of the Tiny ImageNet dataset by selecting 100 classes, taking 500 images per class, and splitting the dataset into training, validation, and test sets.
It also performs image preprocessing (resize, crop) and creates a manifest CSV for each split with information about the images and bounding boxes (if available).
Finally, it calculates normalization statistics and saves them in a JSON file for use in training models like AlexNet
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image

# ---- Constants ----
# I am setting the number of images per class and the split ratio
PER_CLASS = 500  # Total images per class (500)
TRAIN_PER_CLASS = 300  # Training set size per class (300)
VAL_PER_CLASS = 100  # Validation set size per class (100)
TEST_PER_CLASS = 100  # Test set size per class (100)
assert TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS == PER_CLASS, "Split sizes must sum to PER_CLASS"

TARGET_SHORTER_SIDE = 256  # Target shorter side of the image
CROP_SIZE = 224  # Crop size (224x224 for AlexNet input)

# ---- Argument Parsing ----
def parse_args():
    # Parsing arguments from command line to configure paths and options
    parser = argparse.ArgumentParser(description="Build 100x500 Tiny-ImageNet subset + CSV manifests.")
    parser.add_argument("--src", type=Path, required=False,
                        default=Path(r"C:\Users\asmae\Documents\WSU\Deep Learning\assignment2-DL\ques1\tiny-imagenet-200\tiny-imagenet-200\train"),
                        help="Path to Tiny-ImageNet-200 train folder (contains one subfolder per class).")
    parser.add_argument("--dst", type=Path, required=False,
                        default=Path(r"C:\Users\asmae\Documents\WSU\Deep Learning\assignment2-DL\ques1\data_subset_tiny_imagenet"),
                        help="Destination root where {train,val,test}/<class> will be created.")
    parser.add_argument("--random-classes", action="store_true",
                        help="If set, pick 100 classes at random instead of the first 100 sorted.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

# ---- Directory and Class Handling ----
def list_class_dirs(src: Path) -> List[Path]:
    # This function lists all the class directories in the source path Tiny ImageNet.
    # Tiny ImageNet has 200 classes, and this function retrieves all class directories,sorting them alphabetically for consistent class selection.
    return sorted([p for p in src.iterdir() if p.is_dir()])

# ---- This function randomly selects 100 classes from the list of class directories. If 'random_pick' is False, it selects the first 100 classes in sorted order. If 'random_pick' is True, it shuffles the class list and selects 100 classes randomly. 
# And the random seed ensures reproducibility.
def pick_classes(class_dirs: List[Path], random_pick: bool, seed: int) -> List[Path]:
    # Pick 100 classes based on random choice or the first 100 sorted I choose random. 
    if not random_pick:
        return class_dirs[:100]
    rs = random.Random(seed)
    picked = class_dirs.copy()
    rs.shuffle(picked)
    return picked[:100]

# ---- Image Collection ----
# This function collects exactly 500 images for each class from the 'images' subdirectory of the class directory.
# The variable 'PER_CLASS' is defined as 500, and the function ensures there are at least 500 images.
# If there are fewer than 500 images, it raises an error.
# The images are sorted to ensure consistent ordering, and only the first 500 images are returned.
def collect_images_for_class(class_dir: Path) -> List[Path]:
    # Collect exactly 500 images for each class
    images_dir = class_dir / "images"
    imgs = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".jpeg".upper()}])
    if len(imgs) < PER_CLASS:
        raise ValueError(f"{class_dir.name}: expected >= {PER_CLASS} images, found {len(imgs)}")
    return imgs[:PER_CLASS]

# ---- Bounding Box Handling assuming there is only one bounding box per image----
# The function read_bbox_map_if_any reads the bounding box annotations for each image in a class. Each class has a corresponding '<wnid>_boxes.txt' file that contains bounding box data.
def read_bbox_map_if_any(class_dir: Path) -> Dict[str, Tuple[str, str, str, str]]:
    wnid = class_dir.name
    bbox_file = class_dir / f"{wnid}_boxes.txt"
    mapping = {}
    # Here i am checking if bounding box file exists, if not I return empty mapping
    if not bbox_file.exists():
        return mapping
    with open(bbox_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:  # I am making sure I have the correct number of parts which is 5 in total = 1 for the filename + 4 bbox values
                fname, x1, y1, x2, y2 = parts
                mapping[fname] = (x1, y1, x2, y2)  # Here I am mapping image filename directly to bounding box
    return mapping

# ---- Dataset Splitting ----
def split_indices(n: int, train_n: int, val_n: int, test_n: int, seed: int):
    # Split dataset into train, validation, and test based on specified sizes
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    train_idx = idxs[:train_n]
    val_idx = idxs[train_n:train_n + val_n]
    test_idx = idxs[train_n + val_n:train_n + val_n + test_n]
    return train_idx, val_idx, test_idx

# ---- Directory Creation ----
def ensure_dir(p: Path):
    # Ensure that the directory exists
    p.mkdir(parents=True, exist_ok=True)

# ---- Copy and Record Images ----
def copy_and_record(imgs: List[Path], indices: List[int], class_name: str, split: str,
                    dst_root: Path, class_id: int, bbox_map: Dict[str, Tuple[str, str, str, str]],
                    rows_out: List[List[str]]):
    # Copy images to appropriate splits (train, val, test) and record their details in the manifest
    for i in indices:
        src = imgs[i]
        dst = dst_root / split / class_name / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        bbox = bbox_map.get(src.name, ("", "", "", ""))
        rel_path = str(dst.relative_to(dst_root)).replace("\\", "/")
        rows_out.append([split, rel_path, class_name, str(class_id), *bbox])

# ---- Image Preprocessing (Resizing and Cropping) ----
def resize_shorter_side(img: Image.Image, target: int) -> Image.Image:
    # Resize image so the shorter side is equal to the target value
    w, h = img.size
    if min(w, h) == target:
        return img
    if w < h:
        new_w = target
        new_h = int(round(h * (target / w)))
    else:
        new_h = target
        new_w = int(round(w * (target / h)))
    return img.resize((new_w, new_h), Image.BILINEAR)

def center_crop(img: Image.Image, size: int) -> Image.Image:
    # Crop the center of the image to the desired size
    w, h = img.size
    left = (w - size) // 2
    upper = (h - size) // 2
    right = left + size
    lower = upper + size
    return img.crop((left, upper, right, lower))

# ---- Channel Statistics Calculation (for Data Normalization) ----
def compute_channel_stats_on_training(dst_root: Path, class_to_id: Dict[str, int]):
    # Compute mean and std of the images for channel normalization (as done in AlexNet)
    sum_rgb = np.zeros(3, dtype=np.float64)
    sumsq_rgb = np.zeros(3, dtype=np.float64)
    count = 0
    for class_name in class_to_id.keys():
        for img_path in (dst_root / "train" / class_name).glob("*"):
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im = resize_shorter_side(im, 256)
                im = center_crop(im, 224)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                sum_rgb += arr.reshape(-1, 3).sum(axis=0)
                sumsq_rgb += (arr.reshape(-1, 3) ** 2).sum(axis=0)
                count += arr.shape[0] * arr.shape[1]
    mean = (sum_rgb / count).tolist()
    var = (sumsq_rgb / count) - np.square(sum_rgb / count)
    std = np.sqrt(np.maximum(var, 1e-12)).tolist()
    stats = {"mean_rgb": mean, "std_rgb": std, "count_pixels": int(count),
             "resize_shorter_side": 256, "crop_size": 224}
    with open(dst_root / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats

# ---- Manifest Writing ----
def write_label_files(dst_root: Path, picked_classes: List[Path]):
    # Write the class labels to files
    classes = [c.name for c in picked_classes]
    with open(dst_root / "classes.txt", "w") as f:
        for cname in classes:
            f.write(cname + "\n")
    labels = {cname: i for i, cname in enumerate(classes)}
    with open(dst_root / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    return labels

def write_manifests(dst_root: Path, rows_train, rows_val, rows_test):
    # Write the manifest CSV files (train, val, test)
    header = ["split", "relative_path", "class_name", "class_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    for split, rows in [("train", rows_train), ("val", rows_val), ("test", rows_test)]:
        out_csv = dst_root / f"{split}_manifest.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

# ---- Main Function ----
def main():
    # Main function to execute the dataset preparation process
    args = parse_args()  # Parse arguments for paths and options
    random.seed(args.seed)  # Set the random seed for reproducibility

    # Ensure source path exists
    if not args.src.exists():
        raise SystemExit(f"Source path not found: {args.src}")
    
    ensure_dir(args.dst)  # Create destination directories

    class_dirs = list_class_dirs(args.src)  # List all class directories
    if len(class_dirs) < 100:
        raise SystemExit(f"Found {len(class_dirs)} classes; expected at least 100.")
    
    # Pick 100 classes from the dataset (random or first 100)
    picked = pick_classes(class_dirs, args.random_classes, args.seed)

    # Write label files and manifest headers
    labels_map = write_label_files(args.dst, picked)

    rows_train, rows_val, rows_test = [], [], []

    # Process each class
    for class_dir in picked:
        imgs = collect_images_for_class(class_dir)  # Collect images for the class
        bbox_map = read_bbox_map_if_any(class_dir)  # Read bounding box info
        cname = class_dir.name
        cid = labels_map[cname]
        train_idx, val_idx, test_idx = split_indices(len(imgs), TRAIN_PER_CLASS, VAL_PER_CLASS, TEST_PER_CLASS, args.seed)

        # Copy images and record to corresponding split (train/val/test)
        copy_and_record(imgs, train_idx, cname, "train", args.dst, cid, bbox_map, rows_train)
        copy_and_record(imgs, val_idx, cname, "val", args.dst, cid, bbox_map, rows_val)
        copy_and_record(imgs, test_idx, cname, "test", args.dst, cid, bbox_map, rows_test)

    # Write final manifests (CSV files)
    write_manifests(args.dst, rows_train, rows_val, rows_test)

    # Compute channel statistics (mean, std)
    _ = compute_channel_stats_on_training(args.dst, labels_map)

    # Print results
    print("Dataset has been created.")
    print(f"Classes: {len(picked)}  →  saved to: {args.dst}")
    print(f"Splits per class: train={TRAIN_PER_CLASS}, val={VAL_PER_CLASS}, test={TEST_PER_CLASS}")
    print("Files written:")
    print(f"  - {args.dst / 'classes.txt'}")
    print(f"  - {args.dst / 'labels.json'}")
    print(f"  - {args.dst / 'dataset_stats.json'}")
    print(f"  - {args.dst / 'train_manifest.csv'}")
    print(f"  - {args.dst / 'val_manifest.csv'}")
    print(f"  - {args.dst / 'test_manifest.csv'}")

if __name__ == "__main__":
    main()
