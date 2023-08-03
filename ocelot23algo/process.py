from util import gcio
from util.constants import (
    GC_CELL_FPATH, 
    GC_TISSUE_FPATH, 
    GC_METADATA_FPATH,
    GC_DETECTION_OUTPUT_PATH
)

#from user.unet_example.unet import PytorchUnetCellModel
#from user.seg_unet.model import Model
#from user.seg_ensemble_unet.model import Model
#from user.seg_ensemble_yolo.model import Model
from user.seg_ensemble_tta.model import Model

from tqdm import tqdm
from pathlib import Path
import torch

print(f"Torch: {torch.__version__} \n CUDA: {torch.version.cuda} / is available: {torch.cuda.is_available()} \n CUDNN: {torch.backends.cudnn.version()}")
print(f"Loading data from \n `{GC_CELL_FPATH}` Exists: {Path(GC_CELL_FPATH).exists()} \n `{GC_TISSUE_FPATH}` Exists: {Path(GC_TISSUE_FPATH).exists()}")
print(f"Loading metadata from \n `{GC_METADATA_FPATH}` Exists: {Path(GC_METADATA_FPATH).exists()}")

# List directories
print(f"List of directories in `{Path(GC_CELL_FPATH)}`: {list(Path(GC_CELL_FPATH).iterdir())}")
print(f"List of directories in `{Path(GC_TISSUE_FPATH)}`: {list(Path(GC_CELL_FPATH).iterdir())}")

def process():

    #try:
    #    print("Starting debugging")
    #    import debugpy
    #    debugpy.listen(("0.0.0.0", 5678))
    #    debugpy.wait_for_client()
    #except Exception as e:
    #    print(f"Debugging not available: {e}")


    """Process a test patches. This involves iterating over samples,
    inferring and write the cell predictions
    """
    # Initialize the data loader    
    loader = gcio.DataLoader(GC_CELL_FPATH, GC_TISSUE_FPATH)

    # Cell detection writer
    writer = gcio.DetectionWriter(GC_DETECTION_OUTPUT_PATH)
    
    # Loading metadata
    meta_dataset = gcio.read_json(GC_METADATA_FPATH)

    #print(f"{Path(GC_DETECTION_OUTPUT_PATH).parent.exists()}")

    # Instantiate the inferring model
    model = Model(meta_dataset)

    # NOTE: Batch size is 1
    print(f"Start loading images")
    for cell_patch, tissue_patch, pair_id in tqdm(loader):
        print(f"Processing sample pair {pair_id}")
        # Cell-tissue patch pair inference
        cell_classification = model(cell_patch, tissue_patch, pair_id)

        
        # Updating predictions
        writer.add_points(cell_classification, pair_id)

    # Export the prediction into a json file
    print(f"Saving predictions at `{GC_DETECTION_OUTPUT_PATH}` Path exists: {Path(GC_DETECTION_OUTPUT_PATH).parent.exists()}")
    Path(GC_DETECTION_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    writer.save()


if __name__ == "__main__":
    process()

