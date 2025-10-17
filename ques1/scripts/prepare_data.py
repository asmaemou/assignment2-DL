"""
To summarize, this code prepares a subset of the Tiny ImageNet dataset by selecting 100 classes, taking 500 images per class, and splitting the dataset into training, validation, and test sets.
It also performs image preprocessing (resize, crop) and creates a manifest CSV for each split with information about the images and bounding boxes 
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
    parser = argparse.ArgumentParser(description="Build 100x500 Tiny-ImageNet subset + CSV manifests.")
    parser.add_argument("--src", type=Path, required=False,
                        default=Path(r"../../ques1/tiny-imagenet-200/tiny-imagenet-200/train"),
                        help="Path to Tiny-ImageNet-200 train folder ")
    parser.add_argument("--dst", type=Path, required=False,
                        default=Path(r"../data_subset_tiny_imagenet"),
                        help="Destination root where the subset dataset will be created.")
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

def split_indices(n: int, train_n: int, val_n: int, test_n: int, seed: int):
    # This function splits the dataset into training, validation, and test sets based on the specified sizes.
    # I start by creating a list of indices from 0 to n-1,  n represent the total number of samples.
    # Then, I shuffles the indices using a fixed random seed to ensure reproducibility.
    # I divided the dataset is 3 subsets:
    # 'train_n' indices are assigned to the training set, 'val_n' indices are assigned to the validation set and 'test_n' indices are assigned to the test set.
    # Then I split dataset into train, validation, and test 
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    train_idx = idxs[:train_n] # train_idx will get the first train_n image
    val_idx = idxs[train_n:train_n + val_n] #val_idx will get the next val_n images,
    test_idx = idxs[train_n + val_n:train_n + val_n + test_n] # test_idx will get the last test_n images
    return train_idx, val_idx, test_idx

# ---- Directory Creation ----
def ensure_dir(p: Path):
    # Through this function I guarantee that the directories where I will store the subset of Tiny ImageNet images  are created automatically
    p.mkdir(parents=True, exist_ok=True)

# ---- Copy and Record Images ----
def copy_and_record(imgs: List[Path], indices: List[int], class_name: str, split: str,
                    dst_root: Path, class_id: int, bbox_map: Dict[str, Tuple[str, str, str, str]],
                    rows_out: List[List[str]]):
    # In this for loop I loop through the list of indices for the images that need to be copied
    for i in indices:
        # I get the source path of the image based on the current index
        src = imgs[i]
        # Then I create the destination path for the image by combining the destination root, split type, class name, and image name
        dst = dst_root / split / class_name / src.name
        # I ensure that the destination directory exists and creating any missing parent directories if needed.
        # 'parents=True' allows the creation of intermediate directories, and 'exist_ok=True' prevents errors if the directory already exists.
        dst.parent.mkdir(parents=True, exist_ok=True)
        # I copy the image from the source path to the destination path
        shutil.copy2(src, dst)
        # After that I am getting the bounding box data for the image from the bbox_map
        # If no bounding box data is found, return an empty tuple ("", "", "", "")
        bbox = bbox_map.get(src.name, ("", "", "", ""))
        # This line helps generate the file path when preparing my dataset, ensuring that the path will work consistently across different operating systems.
        rel_path = str(dst.relative_to(dst_root)).replace("\\", "/")
        # Finally I append a new row to the rows_out list containing information about the image, including the split type, relative path, class name, class
        rows_out.append([split, rel_path, class_name, str(class_id), *bbox])

# ---- In this function I am resizing the images, in order to prepare the data in a way that aligns with the preprocessing techniques described in the AlexNet paper
def resize_shorter_side(img: Image.Image, target: int) -> Image.Image:
    # Resize the image so that the shorter side matches the target value
    # First I start by getting the current width (w) and height (h) of the image.
    w, h = img.size
    # If the shorter side is  equal to the target value, I don't need to resize, then I just return the original image.
    if min(w, h) == target:
        return img
    # If the width is less than the height  I calculate the new width to be equal to the target value.
    # I calculate the new height by scaling it proportionally based on the original aspect ratio.
    if w < h:
        new_w = target
        new_h = int(round(h * (target / w)))
    # If the height is less than or equal to the width , the new height will be equal to the target value.
    # I then calculate the new width by scaling it proportionally based on the original aspect ratio.
    else:
        new_h = target
        new_w = int(round(w * (target / h)))
    return img.resize((new_w, new_h), Image.BILINEAR)
# ---- This function is used to crop the image in order to prepare the data 
def center_crop(img: Image.Image, size: int) -> Image.Image:
    # Crop the center of the image to the desired size
    w, h = img.size # i am getting the current width and height of the image
    left = (w - size) // 2 # I then calculate the left position for cropping
    upper = (h - size) // 2 # Calculate the upper position for cropping
    right = left + size # I calculate the right position based on the left position and the desired crop size
    lower = upper + size # I calculate the lower position based on the upper position and the desired crop size
    return img.crop((left, upper, right, lower)) # Crop the image and return the cropped image

# ---- Channel Statistics Calculation for Data Normalization ----
def compute_channel_stats_on_training(dst_root: Path, class_to_id: Dict[str, int]):
    # I am computing the mean and std of the images for channel normalization 
    sum_rgb = np.zeros(3, dtype=np.float64)
    sumsq_rgb = np.zeros(3, dtype=np.float64)
    count = 0 # this variable is the total number of pixels processed
    # Loop over each class in the training data
    for class_name in class_to_id.keys():
        for img_path in (dst_root / "train" / class_name).glob("*"): # Iterate over all images in the class
            with Image.open(img_path) as im:
                im = im.convert("RGB") # i need to make sure the image is in RGB format so if it is not i amm converting it 
                im = resize_shorter_side(im, 256) # Resize the shorter side to 256 pixels to match AlexNet preprocessing
                im = center_crop(im, 224) # Center crop to 224x224 pixels
                # Convert the image to a numpy array and normalize pixel values to the range [0, 1]
                arr = np.asarray(im, dtype=np.float32) / 255.0
                # Calculate the sum of RGB values
                sum_rgb += arr.reshape(-1, 3).sum(axis=0)
                # I am calculating the sum of squares of RGB values
                sumsq_rgb += (arr.reshape(-1, 3) ** 2).sum(axis=0)
                count += arr.shape[0] * arr.shape[1]
    mean = (sum_rgb / count).tolist() # i am calculating the mean RGB values
    var = (sumsq_rgb / count) - np.square(sum_rgb / count) # I am calculating the variance of RGB values
    std = np.sqrt(np.maximum(var, 1e-12)).tolist() # I am calculating the standard deviation of RGB values
    stats = {"mean_rgb": mean, "std_rgb": std, "count_pixels": int(count),
             "resize_shorter_side": 256, "crop_size": 224} # i am creating a dictionary to store the computed statistics
    with open(dst_root / "dataset_stats.json", "w") as f: # i am saving the statistics to a JSON file for later use during normalization
        json.dump(stats, f, indent=2)
    return stats
# ---- using this definition i will write class labels to files ----
def write_label_files(dst_root: Path, picked_classes: List[Path]):
    classes = [c.name for c in picked_classes]  # from the picked classes i am creating a list of class names
    with open(dst_root / "classes.txt", "w") as f: # i am writing the class names to a text file named classes.txt
        for cname in classes:
            f.write(cname + "\n") # within each line i will write each class name
    labels = {cname: i for i, cname in enumerate(classes)}  # i am creating a dictionary that maps class names to numerical class IDs
    with open(dst_root / "labels.json", "w") as f: #iam writing the class with the ID in a JSON file named labels.json
        json.dump(labels, f, indent=2)
    return labels
# this function will creates three CSV manifest files, one for each dataset split (train, val, and test).
def write_manifests(dst_root: Path, rows_train, rows_val, rows_test):
    header = ["split", "relative_path", "class_name", "class_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    for split, rows in [("train", rows_train), ("val", rows_val), ("test", rows_test)]:
        out_csv = dst_root / f"{split}_manifest.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
# this is the main function in order prepare the dataset ----
def main():
    args = parse_args()  # here is am parsing arguments for paths and options
    random.seed(args.seed)  # i am setting the random seed for reproducibility
    # Ensure source path exists
    if not args.src.exists():
        raise SystemExit(f"Source path not found: {args.src}")
    ensure_dir(args.dst)  # Create destination directories
    class_dirs = list_class_dirs(args.src)  # List all class directories
    if len(class_dirs) < 100:
        raise SystemExit(f"Found {len(class_dirs)} classes; expected at least 100.")
    picked = pick_classes(class_dirs, args.random_classes, args.seed)  # pick 100 classes from the dataset randomly
    labels_map = write_label_files(args.dst, picked) # write label files and manifest headers
    rows_train, rows_val, rows_test = [], [], []

    # Process each class
    for class_dir in picked:
        imgs = collect_images_for_class(class_dir)  # i am collect exactly 500 images from the 'images' folder of the current class
        bbox_map = read_bbox_map_if_any(class_dir)  # i am reading the bounding box data from the corresponding <wnid>_boxes.txt file
        # i am getting the class name and its corresponding numeric ID
        cname = class_dir.name 
        cid = labels_map[cname]
        train_idx, val_idx, test_idx = split_indices(len(imgs), TRAIN_PER_CLASS, VAL_PER_CLASS, TEST_PER_CLASS, args.seed)
        # I am copying images and record to the corresponding split (train/val/test)
        copy_and_record(imgs, train_idx, cname, "train", args.dst, cid, bbox_map, rows_train)
        copy_and_record(imgs, val_idx, cname, "val", args.dst, cid, bbox_map, rows_val)
        copy_and_record(imgs, test_idx, cname, "test", args.dst, cid, bbox_map, rows_test)

    # i am write final manifests which are CSV files
    write_manifests(args.dst, rows_train, rows_val, rows_test)

    # i am computing the channel statistics mean and std
    _ = compute_channel_stats_on_training(args.dst, labels_map)

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