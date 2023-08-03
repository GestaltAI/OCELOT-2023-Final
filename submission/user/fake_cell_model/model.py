
import os
import numpy as np
import json
from util.constants import SAMPLE_SHAPE


class FakeCellModel():
    """
    U-NET model for cell detection implemented with the Pytorch library

    """
    def __init__(self, metadata):

        print('\033[91' + "!!!! Warning the FakeCellModel is used !!!! \n  Do not use it for submission \n\n")

        gt =  json.load(open("test/fold_0/output/gt.json"))['points']

        self.points = {}
        for item in gt:

            name = item['name']
            if name not in self.points:
                self.points[name] = []

            point = item["point"]
            self.points[name].append((*point, 1.))



    def __call__(self, cell_patch, tissue_patch, pair_id):
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

        image_name = f"image_{pair_id}"
        return self.points[image_name] if image_name in self.points else []
