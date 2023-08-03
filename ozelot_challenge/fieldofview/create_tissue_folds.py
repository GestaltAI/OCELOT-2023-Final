import argparse
from pathlib import Path
import numpy as np
import json

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

parser = argparse.ArgumentParser(description='Create the tissue dataset for MMSegmentation')
parser.add_argument('--input_folder', type=str, 
                    default="/mnt/g/Datasets/ocelot2023_v0.1.2/MMSegTissue_True", 
                    help='Path to the input folder')
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

if __name__ == "__main__":

    meta_data = json.load(open(args.metadata, "r"))

    folder = Path(args.input_folder)

    folds_folder = folder / Path("Folds")
    folds_folder.mkdir(exist_ok=True, parents=True)

    image_names, organ_names, patient_ids = [], [], []

    for image_id, image_meta in meta_data["sample_pairs"].items():

        image_names.append(image_id)
        organ_names.append(organ_to_id[image_meta["organ"]])
        patient_ids.append(image_meta["slide_name"])

    image_names, organ_names, patient_ids = np.array(image_names), np.array(organ_names), np.array(patient_ids)

    kf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    for idx, (train_set, test_set) in enumerate(kf.split(image_names, organ_names,  groups=patient_ids)):

        X_train, X_test = image_names[train_set], image_names[test_set]


        train_paths = [Path(f"{p}.png") for p in X_train]
        test_paths = [Path(f"{p}.png") for p in X_test]

        fold_path = folds_folder / f"fold_{idx}"
        fold_path.mkdir(exist_ok=True, parents=True)
        with open(fold_path / "train.txt", 'w') as f:
            f.writelines(str(line) + '\n' for line in train_paths)

        with open(fold_path / "valid.txt", 'w') as f:
            f.writelines(str(line) + '\n' for line in test_paths)
