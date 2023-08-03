from pathlib import Path
from ultralytics import YOLO
import torch
from torchvision.ops import nms

class YOLOCellModel():

    def __init__(self, metadata:dict) -> None:

        self.metadata = metadata

        self.score_threshold = 0.15
        self.iou_threshold = 0.15


        # Load ensemble models
        _checkpoints_path = Path(__file__).parent / "checkpoints" / "yolo"


        self.models = []
        for model_path in [
            _checkpoints_path / "best_fold_0.pt",
            #_checkpoints_path / "best_fold_1.pt",
        ][:3]:
                        
            self.models.append(YOLO(model_path))


    def prepare_input(self, cell_patch, patch_size=320, patch_overlap=0.75):
        """Preparing the input for the model"""

        # Extract pachtes from the cell patch

        image_height, image_width = cell_patch.shape[:2]

        result_patches, result_coordinates = [], []
        for x in range(0, image_width, int(patch_size * patch_overlap)):
            for y in range(0, image_height, int(patch_size * patch_overlap)):

                
                if x + patch_size > image_width: x = image_width - patch_size
                if y + patch_size > image_height: y = image_height - patch_size

                patch = cell_patch[y:y+patch_size, x:x+patch_size][..., ::-1]  # OpenCV image (BGR to RGB)

                result_patches.append(patch)
                result_coordinates.append((x, y))

        return result_patches, result_coordinates
    
    def post_process(self, bboxes):
        """Post processing the results"""

        predicted_cells = []
        temp_boxes = bboxes["boxes"]
        temp_scores = bboxes["scores"]

        if len(temp_boxes) == 0:
            return predicted_cells

        keep = nms(temp_boxes, temp_scores, self.iou_threshold)

        for box, score, class_id in zip(temp_boxes[keep], temp_scores[keep], bboxes["classes"][keep]):

            xc, yc, = box[0] + (box[2] - box[0]) / 2, box[1] + (box[3] - box[1]) / 2
            predicted_cells.append((round(float(xc)), round(float(yc)), int(class_id) + 1, float(score)))

        return predicted_cells
    
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

        results = {
            "boxes" : [],
            "scores" : [],
            "classes" : []
        }
        cell_patches, patch_coordinates = self.prepare_input(cell_patch)

        for cell_patch, coords in zip(cell_patches, patch_coordinates):
            x_min, y_min = coords

            for model in self.models:

                pred = model.predict(cell_patch, stream=False, save=False, conf=self.score_threshold, verbose=False)[0].cpu()

                for box, score, class_id in zip(pred.boxes.xyxy, pred.boxes.conf, pred.boxes.cls): #, pred.masks.segments
                    
                    results["boxes"].append(box + torch.tensor((x_min, y_min, x_min, y_min)))
                    results["scores"].append(score)
                    results["classes"].append(class_id)
    
        results["boxes"] = torch.stack(results["boxes"]) if len(results["boxes"]) > 0 else torch.tensor([])
        results["scores"] = torch.stack(results["scores"]) if len(results["scores"]) > 0 else torch.tensor([])
        results["classes"] = torch.stack(results["classes"]) if len(results["classes"]) > 0 else torch.tensor([])

        return self.post_process(results)