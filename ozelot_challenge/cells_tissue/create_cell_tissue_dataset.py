import argparse
from pathlib import Path
import numpy as np
import json
import cv2
from PIL import Image as PILImage

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Create the tissue dataset for MMSegmentation')
parser.add_argument('--input_folder', type=str, 
                    default="/mnt/g/Datasets/ocelot2023_v0.1.2/", 
                    help='Path to the input folder')
parser.add_argument('--metadata', type=str, 
                    default="/mnt/g/Datasets/ocelot2023_v0.1.2/metadata.json", 
                    help='Path to the metadata file')
parser.add_argument('--include_unknown', type=bool, default=True, 
                    help='Include unknown class or assign to background label')
args = parser.parse_args()


if __name__ == "__main__":

    meta_data = json.load(open(args.metadata, "r"))
    include_unknwon = args.include_unknown

    folder = Path(args.input_folder)

    for image_id, image_meta in tqdm(meta_data["sample_pairs"].items(), total=len(meta_data["sample_pairs"])):

        tissue_path = folder / "images" / "train" / "tissue" / f"{image_id}.jpg"
        tissue_mask_path = folder / "annotations" / "train" / "tissue" / f"{image_id}.png"

        mask_image = PILImage.open(tissue_mask_path)
        mask = np.array(mask_image)

        # extract the cell patch from the tissue mask
        cell_x_start = image_meta["cell"]["x_start"]
        cell_y_start = image_meta["cell"]["y_start"]

        cell_x_end = image_meta["cell"]["x_end"]
        cell_y_end = image_meta["cell"]["y_end"]


        tissue_x_start = image_meta["tissue"]["x_start"]
        tissue_y_start = image_meta["tissue"]["y_start"]

        tissue_x_end = image_meta["tissue"]["x_end"]
        tissue_y_end = image_meta["tissue"]["y_end"]

        # compute the relative coordinates of the cell patch with respect to the tissue patch
        mask_x_start_rel = (cell_x_start-tissue_x_start) / (tissue_x_end-tissue_x_start)
        mask_y_start_rel = (cell_y_start-tissue_y_start) / (tissue_y_end-tissue_y_start)

        mask_x_end_rel = (cell_x_end-tissue_x_start) / (tissue_x_end-tissue_x_start)
        mask_y_end_rel = (cell_y_end-tissue_y_start) / (tissue_y_end-tissue_y_start)

        # Compute the absolute coordinates of the cell patch with respect to the tissue patch
        mask_x_start_abs = round(mask_x_start_rel * mask_image.width)
        mask_y_start_abs = round(mask_y_start_rel * mask_image.height)

        mask_x_end_abs = round(mask_x_end_rel * mask_image.width)
        mask_y_end_abs = round(mask_y_end_rel * mask_image.height)

        # create the cell patch mask
        cell_mask = mask[mask_y_start_abs:mask_y_end_abs, mask_x_start_abs:mask_x_end_abs]
        cell_image_mask = PILImage.fromarray(cell_mask)

        cell_path = folder / "images" / "train" / "cell" / f"{image_id}.jpg"
        cell_image = PILImage.open(cell_path)

        # resize the cell image to the size of the cell patch mask
        cell_image_mask = np.array(cell_image_mask.resize(cell_image.size, PILImage.Resampling.NEAREST))

        # 0: background, 1: tissue, 2: unknown
        cell_image_mask -= 1
        # If include unknown, set unknown to 2 (seperate label), else to 0 (background)
        if include_unknwon:
            cell_image_mask[cell_image_mask == 254] = 2
        else:            
            cell_image_mask[cell_image_mask == 254] = 0


        # Create the folder structure
        result_folder = folder / Path(f"MMSegCells_{include_unknwon}")

        # Training images
        target_folder = result_folder / "img_dir" / "train"
        target_folder.mkdir(exist_ok=True, parents=True)

        # Training masks
        mask_target_folder = result_folder / "ann_dir" / "train"
        mask_target_folder.mkdir(exist_ok=True, parents=True)

        # Copy the training image
        cell_image.save(str(target_folder / f"{image_id}.png"))

        #Save the mask image
        cv2.imwrite(str(mask_target_folder / f"{image_id}.png"), cell_image_mask)




