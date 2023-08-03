import numpy as np
from PIL import Image as PILImage

from .mmseg import SegmentationInference
from .unet import PytorchUnetCellModel  #user.seg_unet_2class
#from ..fake_cell_model.model import FakeCellModel as PytorchUnetCellModel  #user.seg_unet_2class


class Model():
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.metadata = metadata

        self._cell_model = PytorchUnetCellModel(metadata)
        self._tissue_model = SegmentationInference(metadata)


    def post_process(self, cell_predictions, mask:PILImage):
        """This function relabels cell predictions based on the tissue mask.
            
        Parameters
        ----------
        cell_predictions: List[tuple]
            for each predicted cell we provide the tuple (x, y, cls, score)
        mask: PILImage
            Each pixel corresponds to the cell type label. 
            1 means background
            2 means tumor
            3 means Unknown

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """

        np_mask = np.array(mask)
        relabelt = []
        for xs, ys, class_id, probs in cell_predictions:

            x_min = int(max(xs - 5, 0))
            y_min = int(max(ys - 5, 0))
            x_max = int(min(xs + 5, mask.width))
            y_max = int(min(ys + 5, mask.height))

            # crop the mask to the size of the input image
            mask_crop = np_mask[y_min:y_max, x_min:x_max]
            new_class_id = int(np.median(mask_crop))

            # If the new class ID is 3, then we use the old class ID
            if new_class_id == 3: new_class_id = class_id

            relabelt.append((xs, ys, new_class_id, probs))  #new_class_id
        return relabelt

    def __call__(self, cell_patch, tissue_patch, pair_id):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided. 

        NOTE: this implementation offers a dummy inference example. This must be
        updated by the participant.

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
        # Getting the metadata corresponding to the patch pair ID
        meta_pair = self.metadata[pair_id]

        #############################################
        #### YOUR INFERENCE ALGORITHM GOES HERE #####
        #############################################

        # Call the tissue model first 
        tissue_prediction = self._tissue_model(cell_patch, tissue_patch, pair_id)

        # List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        cell_predictions = self._cell_model(cell_patch, tissue_patch, pair_id)

        results = self.post_process(cell_predictions, tissue_prediction)


        #############################################
        ####### RETURN RESULS PER SAMPLE ############
        #############################################

        # We need to return a list of tuples with 4 elements, i.e.:
        # - int: cell's x-coordinate in the cell patch
        # - int: cell's y-coordinate in the cell patch
        # - int: class id of the cell, either 1 (BC) or 2 (TC)
        # - float: confidence score of the predicted cell
        # list(zip(xs, ys, class_id, probs))
        return results
