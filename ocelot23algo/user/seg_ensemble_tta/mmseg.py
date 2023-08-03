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

        # Load ensemble models
        _checkpoints_path = Path(__file__).parent / "checkpoints"

        self.models = []
        for model_path, cfg_path in [            
            (_checkpoints_path / "segformer_b2/fold_0.pth", _checkpoints_path / "segformer_b2/fold_0_config.py"),
            (_checkpoints_path / "segformer_b2/fold_1.pth", _checkpoints_path / "segformer_b2/fold_1_config.py"),
            (_checkpoints_path / "segformer_b2/fold_2.pth", _checkpoints_path / "segformer_b2/fold_2_config.py"),
            (_checkpoints_path / "segformer_b2/fold_3.pth", _checkpoints_path / "segformer_b2/fold_3_config.py"),
            (_checkpoints_path / "segformer_b2/fold_4.pth", _checkpoints_path / "segformer_b2/fold_4_config.py"),
        ][:]:
            self.models.append(self.load_checkpoint(model_path, cfg_path))
            

    def load_checkpoint(self, model_path, cfg_path, tta=False):

        _cfg = Config.fromfile(cfg_path)

        if tta:
            _cfg.test_dataloader.dataset.pipeline = _cfg.tta_pipeline
            _cfg.tta_model.module = _cfg.model
            _cfg.model = _cfg.tta_model


        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = init_model(_cfg, str(model_path), device) #'cuda:0'


        return model

    def prepare_input(self, patch):
        """Preparing the input for the model"""
        return patch

    def post_process(self, results, image_meta):
        """Post processing the results"""

        # Sum up the activations of the three models
        m = nn.Softmax(dim = 0)
        logits_mask = results[0].seg_logits.data
        for i in range(1, len(results)):
            logits_mask += results[i].seg_logits.data

        # Compute the softmax mask of the three models
        softmax_mask = m(logits_mask)

        # Add the 1 to the mask to make it compatible with the cell type labels
        mask = softmax_mask.argmax(dim=0).cpu().numpy() + 1

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
        mask_x_start_abs = round(mask_x_start_rel * results[0].img_shape[0])    #width?
        mask_y_start_abs = round(mask_y_start_rel * results[0].img_shape[1])    #height?

        mask_x_end_abs = round(mask_x_end_rel * results[0].img_shape[0])      #width?
        mask_y_end_abs = round(mask_y_end_rel * results[0].img_shape[1])    #height?

        # create the cell patch mask
        cell_mask = mask[mask_y_start_abs:mask_y_end_abs, mask_x_start_abs:mask_x_end_abs]

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
        results = []
        #img = mmcv.imread(tissue_patch)  # OpenCV image (BGR to RGB)
        #img = self.prepare_input(img)

        #results = [inference_model(model, img) for model in self.models]

        # TTA  OpenCV image (BGR to RGB)
        img = mmcv.imread(tissue_patch) #[..., ::-1]  # OpenCV image (BGR to RGB)
        img = self.prepare_input(img)
        results.extend([inference_model(model, img) for model in self.models])

        return self.post_process(results, self.metadata[pair_id])
