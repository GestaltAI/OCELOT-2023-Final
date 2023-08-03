import numpy as np
from user.seg_ensemble_tta.model import SubModel

class Model():
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.metadata = metadata
        self.model = SubModel(self.metadata)

    def __call__(self, cell_patch, tissue_patch, pair_id):
        return self.model(cell_patch, tissue_patch, pair_id)
