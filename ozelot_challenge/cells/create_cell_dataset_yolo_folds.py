import argparse
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
from PIL import Image as PILImage
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

parser = argparse.ArgumentParser(description='Create the tissue dataset for MMSegmentation')
parser.add_argument('--output_folder', type=str, 
                    default="/mnt/g/Datasets/ocelot2023_v0.1.2/MMSegCellsYolo", 
                    help='Path to the ouput folder')
parser.add_argument('--metadata', type=str, 
                    default="/mnt/g/Datasets/ocelot2023_v0.1.2/metadata.json", 
                    help='Path to the metadata file')

args = parser.parse_args()

organ_to_id = {
    "head-and-neck": 0,
    "kidney": 1,
    "endometrium": 2,
    "bladder": 3,
    "prostate": 4,
    "stomach": 5,
}

def create_yolo_dataset(input_folder, ids, images_folder, annotations_folder, cell_size:int=30):

    images_folder.mkdir(exist_ok=True, parents=True)
    annotations_folder.mkdir(exist_ok=True, parents=True)

    for id in tqdm(ids):

        image_path = input_folder / "images" / "train" / "cell" / f"{id}.jpg"
        image = PILImage.open(str(image_path))

        annotation_path = input_folder / "annotations" / "train" / "cell" / f"{id}.csv"

        df = pd.read_csv(annotation_path, names=["x", "y", "tumor"])

        df["x"] = df["x"] / image.width
        df["y"] = df["y"] / image.height

        cell_width = cell_size / image.width
        cell_height = cell_size / image.height

        rows = []
        for _, row in df.iterrows():

            x, y = row["x"], row["y"]
            tumor = int(row["tumor"] - 1)

            rows.append([tumor, x, y, cell_width, cell_height])

        image.save(str(images_folder / f"{id}.png"))
        with open(str(annotations_folder / f"{id}.txt"), "w") as f:
            f.write("\n".join([" ".join(map(str, row_)) for row_ in rows]))

if __name__ == "__main__":

    meta_data = json.load(open(args.metadata, "r"))

    input_folder = Path(args.metadata).parent

    output_folder = Path(args.output_folder)

    folds_folder = output_folder / Path("Folds")
    folds_folder.mkdir(exist_ok=True, parents=True)

    image_names, organ_names, patient_ids = [], [], []

    for image_id, image_meta in meta_data["sample_pairs"].items():

        image_names.append(image_id)
        organ_names.append(organ_to_id[image_meta["organ"]])
        patient_ids.append(image_meta["slide_name"])

    image_names, organ_names, patient_ids = np.array(image_names), np.array(organ_names), np.array(patient_ids)

    kf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    for i, (train_set, test_set) in enumerate(kf.split(image_names, organ_names,  groups=patient_ids)):

        X_train, X_test = image_names[train_set], image_names[test_set]

        fold =  f"fold_{i}"

        train_images_folder = folds_folder / fold / 'images/train/'
        val_images_folder = folds_folder / fold / 'images/val/'
        train_annotations_folder = folds_folder / fold /  'labels/train/'
        val_annotations_folder = folds_folder / fold /  'labels/val/'

        create_yolo_dataset(input_folder, X_train, train_images_folder, train_annotations_folder)
        create_yolo_dataset(input_folder, X_test, val_images_folder, val_annotations_folder)