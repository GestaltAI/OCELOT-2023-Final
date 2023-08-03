from util import gcio
from util.constants import (
    GC_CELL_FPATH, 
    GC_TISSUE_FPATH, 
    GC_METADATA_FPATH,
    GC_DETECTION_OUTPUT_PATH
)

from user.unet_example.unet import PytorchUnetCellModel
from user.seg_unet.model import Model
#from user.seg_ensemble_unet.model import Model
#from user.seg_ensemble_yolo.model import Model
from user.seg_unet_tta.model import Model

from tqdm import tqdm
from pathlib import Path

def process():
    """Process a test patches. This involves iterating over samples,
    inferring and write the cell predictions
    """
    # Initialize the data loader
    loader = gcio.DataLoader(GC_CELL_FPATH, GC_TISSUE_FPATH)

    # Cell detection writer
    writer = gcio.DetectionWriter(GC_DETECTION_OUTPUT_PATH)
    
    # Loading metadata
    meta_dataset = gcio.read_json(GC_METADATA_FPATH)

    # Instantiate the inferring model
    model = Model(meta_dataset)

    # NOTE: Batch size is 1
    for cell_patch, tissue_patch, pair_id in tqdm(loader):
        #print(f"Processing sample pair {pair_id}")
        # Cell-tissue patch pair inference
        cell_classification = model(cell_patch, tissue_patch, pair_id)

        
        # Updating predictions
        writer.add_points(cell_classification, pair_id)

    # Export the prediction into a json file
    writer.save()


if __name__ == "__main__":
    process()
