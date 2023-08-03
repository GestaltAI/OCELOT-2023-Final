from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import shutil
from PIL import Image as PILImage
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

parser = argparse.ArgumentParser(description='Create the tissue dataset for MMSegmentation')
parser.add_argument('--input_folder', type=str, 
                    default="/mnt/g/Datasets/ocelot2023_v0.1.2", 
                    help='Path to the input folder')
parser.add_argument('--include_unknown', type=bool, default=False, 
                    help='Include unknown class or assign to background label')
args = parser.parse_args()

if __name__ == "__main__":

    input_folder = Path(args.input_folder)
    include_unknown = False #args.include_unknown

    # Create the folder structure
    result_folder = input_folder / Path(f"MMSegTissue_{include_unknown}")

    # Training images
    target_folder = result_folder / "img_dir" / "train"
    target_folder.mkdir(exist_ok=True, parents=True)

    # Training masks
    mask_target_folder = result_folder / "ann_dir" / "train"
    mask_target_folder.mkdir(exist_ok=True, parents=True)

    for image_path in tqdm(list((input_folder / "images" / "train" / "tissue").glob("*.jpg"))):
        # Check if the image exists
        if not image_path.exists():
            raise ValueError(f"Image {image_path} does not exist")

        # Create the path for the mask image and check if it exists
        mask_path = input_folder / "annotations" / "train" / "tissue" / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise ValueError(f"Mask image {mask_path} does not exist")

        # 1: BG, 2: tissue, 255: unknown
        mask_image = np.array(PILImage.open(str(mask_path)))

        # If include unknown, set unknown to 2 (seperate label), else to 0 (background)
        if include_unknown:
            mask_image[mask_image == 254] = 2
        else:            
            mask_image[mask_image == 254] = 0

        # Copy the training image
        PILImage.open(str(image_path)).save(str(target_folder / mask_path.name))

        #Save the mask image
        cv2.imwrite(str(mask_target_folder / mask_path.name), mask_image)





    
