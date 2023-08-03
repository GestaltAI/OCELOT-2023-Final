from pathlib import Path

from PIL import Image as PILImage
from mmengine.config import Config
from mmseg.apis import init_model, inference_model
import mmcv
import numpy as np
import torch
import torch.nn as nn

class SegmentationInference():

    def __init__(self, metadata:dict) -> None:

        self.metadata = metadata
        self.model = self.load_checkpoint()

    def load_checkpoint(self):
        """Loading the trained weights to be used for validation"""
        _curr_path = Path(__file__).parent / "checkpoints" / "segformer_b2" 


        _cfg_path = _curr_path / "config.py"
        _path_to_checkpoint = _curr_path / "iter_20000.pth"  #Fold 0: F1:  0.65565
        
        #_cfg_path = _curr_path / "segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py"
        #_path_to_checkpoint = _curr_path / "b0_iter_60000.pth"  #Fold 0: F1: 0.58


        #_cfg_path = _curr_path / "segformer_mit-b1_8xb1-160k_cityscapes-1024x1024.py"
        #_path_to_checkpoint = _curr_path / "b1_iter_32000.pth"   #Fold 0: F1: 0.65675


        _cfg = Config.fromfile(_curr_path / _cfg_path)

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = init_model(_cfg, str(_path_to_checkpoint), device) #'cuda:0'
        return model

    def prepare_input(self, patch):
        """Preparing the input for the model"""
        return patch

    def post_process(self, result, image_meta):
        """Post processing the results"""

        # Add the 1 to the mask to make it compatible with the cell type labels
        mask = result.pred_sem_seg.data.cpu().numpy() + 1
        # 
        #m = nn.Softmax(dim = 0)
        #sm = m(result.seg_logits.data)
        #label_mask = m(result.seg_logits.data).argmax(dim=0).cpu().numpy() + 1

        # Crop the mask to the size of the input image
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
        mask_x_start_abs = round(mask_x_start_rel * result.img_shape[0])    #width?
        mask_y_start_abs = round(mask_y_start_rel * result.img_shape[1])    #height?

        mask_x_end_abs = round(mask_x_end_rel * result.img_shape[0])      #width?
        mask_y_end_abs = round(mask_y_end_rel * result.img_shape[1])    #height?

        # create the cell patch mask
        cell_mask = mask[0][mask_y_start_abs:mask_y_end_abs, mask_x_start_abs:mask_x_end_abs]

        return PILImage.fromarray(cell_mask.astype(np.uint8)).resize((1024, 1024), PILImage.Resampling.NEAREST)


    def __call__(self, cell_patch, tissue_patch, pair_id) -> PILImage:
        """This function detects the cells in the cell patch using Pytorch U-Net.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8] 
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """

        img = mmcv.imread(tissue_patch)
        img = self.prepare_input(img)

        result = inference_model(self.model, img)

        return self.post_process(result, self.metadata[pair_id])
