from pathlib import Path
import argparse
import numpy as np
from PIL import Image as PILImage
import pandas as pd
import json
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Create the evaluation dataset for the docker container')
parser.add_argument('--input_folder', type=str, 
                    default="/mnt/g/Datasets/ocelot2023_v0.1.2", 
                    help='Path to the input folder')
parser.add_argument('--output_folder', type=str, 
                    default="/mnt/c/ProgProjekte/Python/OCELOTMICCAI23/ocelot23algo/test/fold_0", 
                    help='Path to the output folder to save the stacked images and GT')
parser.add_argument('--val_images_list_path', type=str, 
                    default="/mnt/g/Datasets/ocelot2023_v0.1.2/MMSegTissue_True/Folds/fold_0/valid.txt", 
                    help='Path to text file containing the validation images')
args = parser.parse_args()

if __name__ == "__main__":


    input_folder = Path(args.input_folder)
    meta_data = json.load(open(str(input_folder / "metadata.json"), "r"))

    with open(args.val_images_list_path) as file:
        val_ids = [line.rstrip() for line in file]

    # Create the folder structure
    result_folder_tissue = Path(args.output_folder) / "input" / "images" / "tissue_patches"
    result_folder_tissue.mkdir(exist_ok=True, parents=True)

    result_folder_cells = Path(args.output_folder) / "input" / "images" / "cell_patches"
    result_folder_cells.mkdir(exist_ok=True, parents=True)
    
    result_metadata_json = Path(args.output_folder) / "input" / "metadata.json"
    result_gt_json = Path(args.output_folder) / "output" / "gt.json"
    result_gt_json.parent.mkdir(exist_ok=True, parents=True)


    new_metadata = []
    tissue_stack = []
    cell_stack = []
    gt_data = {
            "type": "Multiple points",
            "points": [],
            "version": {"major": 1, "minor": 0},
        }
    # Convert the images

    for idx, val_id in tqdm(enumerate(val_ids), total=len(val_ids)):

        val_id = val_id.split(".")[0]# Convert 264.png to 264

        image_meta = meta_data["sample_pairs"][val_id]

        tissue_image = PILImage.open(str(input_folder / "images" / "train" / "tissue" / f"{val_id}.jpg"))
        cell_image = PILImage.open(str(input_folder / "images" / "train" / "cell" / f"{val_id}.jpg"))

        tissue_stack.append(np.array(tissue_image))
        cell_stack.append(np.array(cell_image))

        gt_df = pd.read_csv(str(input_folder / "annotations" / "train" / "cell" / f"{val_id}.csv"), names=["x", "y", "tumor"])
        for _, row in gt_df.iterrows():

            gt_data["points"].append(
                {
                    "name": f"image_{idx}",
                    "point": [int(row["x"]), int(row["y"]), int(row["tumor"])],
                    "probability": 1.0
                }
            )
        new_metadata.append(image_meta)


    gt_data["num_images"] = len(val_ids)

    with open(str(result_gt_json), "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=4)


    with open(str(result_metadata_json), "w", encoding="utf-8") as f:
        json.dump(new_metadata, f, ensure_ascii=False, indent=4)

    tissue_stack = np.vstack(tissue_stack)
    cell_stack = np.vstack(cell_stack)

    PILImage.fromarray(tissue_stack).save(result_folder_tissue / 'tissue.tif')
    PILImage.fromarray(cell_stack).save(result_folder_cells / 'cells.tif')