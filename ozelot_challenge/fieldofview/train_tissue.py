from typing import List, Sequence
from pathlib import Path
import wandb
from tqdm import tqdm

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.device import get_device
from mmengine.runner import Runner
from mmengine.evaluator import BaseMetric

from mmseg.registry import DATASETS, METRICS
from mmseg.datasets import BaseSegDataset

import pandas as pd
import json
import mim

CLASSES = (
           #"UNKOWN",   # 0
           "BG", # 1
           "TUMOR", # 2
        )


PALETTE = [
    #[0,0,0],   #"UNKOWN",   # 0
    [0,255,0], #"BG", # 1
    [255,0,0]  # "TUMOR", # 2
]


@DATASETS.register_module()
class OZELOTDataset(BaseSegDataset):
  METAINFO = dict(classes = CLASSES, palette = PALETTE)
  def __init__(self, reduce_zero_label:bool=True, **kwargs):
    super().__init__(img_suffix='', reduce_zero_label=reduce_zero_label, seg_map_suffix='', **kwargs)


from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from typing import Dict, List, Optional, Sequence
@METRICS.register_module()
class F1_OCELOTSegmentationScoreMetric(BaseMetric):
    def __init__(self, csv_annotation_folder: str, metadata_path: str, 
                 collect_device: str = 'cpu', prefix: Optional[str] = 'OCELOT'):
        """
        The metric first processes each batch of data_samples and predictions,
        and appends the processed results to the results list. Then it
        collects all results together from all ranks if distributed training
        is used. Finally, it computes the metrics of the entire dataset.
        """
        super().__init__(collect_device=collect_device, prefix=prefix)

        csv_annotation_folder = Path(csv_annotation_folder) if isinstance(csv_annotation_folder, str) else csv_annotation_folder
        self.gt_dfs = {p.stem: pd.read_csv(p, names=["x", "y", "tumor"]) for p in tqdm(list(csv_annotation_folder.glob("*.csv")), desc="Loading GT for the F1-Metric")}

        self.meta_data = json.load(open(metadata_path, "r"))["sample_pairs"]

        

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        
        for data_sample  in data_samples:

            gt_id = Path(data_sample["img_path"]).stem
            df = self.gt_dfs[gt_id]

            # Continue, if the file has no ground truth cells
            if len(df) == 0: continue

            image_meta = self.meta_data[gt_id]

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
            mask_x_start_abs = round(mask_x_start_rel * 1024) # Todo: replace 1024 with the actual size of the tissue patch
            mask_y_start_abs = round(mask_y_start_rel * 1024) # Todo: replace 1024 with the actual size of the tissue patch

            mask_x_end_abs = round(mask_x_end_rel * 1024) # Todo: replace 1024 with the actual size of the tissue patch
            mask_y_end_abs = round(mask_y_end_rel * 1024) # Todo: replace 1024 with the actual size of the tissue patch

            # Campute the with and height and the cell patch with respect to the tissue patch
            mask_width_rel  = 1024 / (mask_x_end_abs - mask_x_start_abs)
            mask_height_rel = 1024 / (mask_y_end_abs - mask_y_start_abs)


            num_tp_bg = 0
            num_fp_bg = 0
            num_gt_bg = 0

            num_tp_tc = 0
            num_fp_tc = 0
            num_gt_tc = 0

            # Compute the absolute coordinates of the cell patch with respect to the tissue patch
            for _, row in df.iterrows():

                x, y = mask_x_start_abs + round(row["x"] / mask_width_rel), mask_y_start_abs + round(row["y"] / mask_height_rel)
                gt_label = int(row["tumor"])

                if gt_label == 0:
                    raise ValueError("The label of the ground truth is 0, which is not allowed. Please check the csv file.")

                # extract the label of the predicted segmentation in an area of two pixels around the cell
                slice = data_sample["pred_sem_seg"]["data"][0, max(0, y-2):min(1024, y+2), max(0, x-2): min(1024, x+2)]
                if slice.shape[0] == 0 or slice.shape[1] == 0:
                    raise ValueError("The size of the slice is 0. Please check the csv file.")

                pred_label = int(slice.median()) + 1


                # Compute the image statistics for the background cell
                if gt_label == 1:
                    num_gt_bg += 1
                    if pred_label == 1:
                        num_tp_bg += 1
                    else:
                        num_fp_bg += 1

                # Compute the image statistics for the tumor cell
                if gt_label == 2:
                    num_gt_tc += 1
                    if pred_label == 2:
                        num_tp_tc += 1
                    else:
                        num_fp_tc += 1


            self.results.append((num_tp_bg, num_fp_bg, num_gt_bg, num_tp_tc, num_fp_tc, num_gt_tc))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """

        global_num_tp_bg, global_num_fp_bg, global_num_gt_bg, global_num_tp_tc, global_num_fp_tc, global_num_gt_tc = np.sum(np.array(results), axis=0)

        precision_bg = global_num_tp_bg / (global_num_tp_bg + global_num_fp_bg + 1e-7)
        recall_bg = global_num_tp_bg / (global_num_gt_bg + 1e-7)
        f1_bg = 2 * precision_bg * recall_bg / (precision_bg + recall_bg + 1e-7)

        precision_tc = global_num_tp_tc / (global_num_tp_tc + global_num_fp_tc + 1e-7)
        recall_tc = global_num_tp_tc / (global_num_gt_tc + 1e-7)
        f1_tc = 2 * precision_tc * recall_tc / (precision_tc + recall_tc + 1e-7)

        metrics = {
            'mF1': (f1_bg + f1_tc) / 2,

            'Pre/BC': precision_bg,
            'Rec/BC': recall_bg,
            'F1/BC': f1_bg,
            'Pre/TC': precision_tc,
            'Rec/TC': recall_tc,
            'F1/TC': f1_tc,

        }
        return metrics


num_classes = 3
bs = 1
dataset_type = 'OZELOTDataset'

model_type = "SegFormer" # deeplabv3plus  SegFormer

if model_type == "SegFormer":
    network_size = 1
    experiment_name = f"SegFormer_B{network_size}_IU_BS{bs}"
    model_name = f"segformer_mit-b{network_size}_8xb1-160k_cityscapes-1024x1024" #b0, b1 (6), b2, b3, b4, b5
elif model_type == "deeplabv3plus":

    network_size = "r50b"  # r18b (16), r50b (5), r101b ()  # 24GB
    experiment_name = f"Deeplabv3plus_{network_size}_IU_BS{bs}"   
    model_name = f"deeplabv3plus_{network_size}-d8_4xb2-80k_cityscapes-769x769"

# https://drive.google.com/file/d/18d2F3bfyODUeNqflvDDiZzrhd47prTY6/view?usp=sharing
data_root = '/host_Data/Datasets/MICCAI23-OcelotChallenge/MMSegTissue_False'
if Path(data_root).exists() == False:
    data_root = "/media/bronzi/00FC6605FC65F4F6/Datasets/OZELOT/MMSegTissue_False"

# https://drive.google.com/file/d/120-hKUhmXDzcc18OgQuD2Z-THCIy4JWP/view?usp=sharing
csv_annotations_gt = '/host_Data/Datasets/MICCAI23-OcelotChallenge/GT/annotations/train/cell'
if Path(csv_annotations_gt).exists() == False:
    csv_annotations_gt = "Data/annotations/train/cell"

# https://drive.google.com/file/d/120-hKUhmXDzcc18OgQuD2Z-THCIy4JWP/view?usp=sharing
metadata_path = "/host_Data/Datasets/MICCAI23-OcelotChallenge/GT/metadata.json"
if Path(metadata_path).exists() == False:
    metadata_path = "Data/metadata.json"

img_dir = 'img_dir/train'
ann_dir = 'ann_dir/train'
train_split = 'Folds/fold_0/train.txt'
val_split = 'Folds/fold_0/valid.txt'
work_dir = 'logs/tutorial'
max_iters = 20000
val_interval = 250

# Download the model config and weights
ckp_path = mim.download(package="mmsegmentation", configs=[model_name], dest_root="models")[0]
Path(work_dir).mkdir(exist_ok=True, parents=True)


cfg = Config.fromfile(f"models/{model_name}.py")
#cfg.compile = True
cfg.amp = True

#cfg.auto-scale-lr

# Set seed to facitate reproducing the result
cfg["experiment_name"] = experiment_name
cfg['randomness'] = dict(seed=0)
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = get_device()


# Modify dataset type and path
cfg.dataset_type = dataset_type
cfg.data_root = data_root

cfg.model.backbone.init_cfg = None # Remove the pretrained weights from the backbone
cfg.load_from = f"models/{ckp_path}" # Load the pre-trained model including the backbone weights


# Gut feeling that GN works better than BN. !Check!
#cfg.model.decode_head.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
cfg.model.decode_head.num_classes = num_classes
#cfg.model.decode_head.ignore_index = 0 # Ignore index zero for loss calculation (unlabeld data)

class_weights = [
  1., #"BG"
  1., #"TUMOR"
  1., #"UNKOWN"
]
cfg.model.decode_head.loss_decode = [
                                        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=class_weights, avg_non_ignore=True ),
                                        #dict(type='LovaszLoss', loss_name='loss_lovasz', loss_weight=3.0, reduction='none', class_weight=class_weights),
                                        #dict(type='FocalLoss', loss_name='focal_loss', loss_weight=1.0, reduction='mean', class_weight=class_weights)
                                    ]

# RandomCutOut, RandomMosaic
# https://github.com/open-mmlab/mmsegmentation/blob/6c3e63e48b8f94a64ac5b1d88cab6fca005ae269/mmseg/datasets/pipelines/__init__.py
# RandomMosaic?
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    #dict(type='RandomMosaic', prob=0.2), #https://github.com/open-mmlab/mmsegmentation/blob/e64548fda0221ad708f5da29dc907e51a644c345/docs/zh_cn/advanced_guides/add_datasets.md
    #dict(
    #    type='RandomResize',
    #    scale=(512, 1024),
    #    ratio_range=(0.5, 1.0),
    #    keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='BioMedicalRandomGamma', prob=0.5),
    dict(type='RandomFlip', direction="vertical", prob=0.5),
    dict(type='RandomRotate', degree=(-45, 45), prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
cfg.tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in [1]
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
                dict(type='RandomFlip', prob=0., direction='vertical'),
                dict(type='RandomFlip', prob=1., direction='vertical')
            ], 
            [dict(type='LoadAnnotations')], 
        ])
]


cfg.train_dataloader.batch_size = bs
cfg.train_dataloader.num_workers = bs * 2
cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.train_dataloader.dataset.pipeline = train_pipeline
cfg.train_dataloader.dataset.ann_file = train_split

cfg.val_dataloader.batch_size = 1
#cfg.val_dataloader.num_workers = bs * 2
cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
cfg.val_dataloader.dataset.ann_file = val_split

cfg.test_dataloader.batch_size = 1
#cfg.test_dataloader.num_workers = bs * 2
cfg.test_dataloader.dataset.type = cfg.dataset_type
cfg.test_dataloader.dataset.data_root = cfg.data_root
cfg.test_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
cfg.test_dataloader.dataset.ann_file = val_split


# Set up working dir to save files and logs.
cfg.work_dir = work_dir

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
if cfg.amp:
    cfg.optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer, clip_grad=None, loss_scale='dynamic')
else:
    cfg.optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

cfg.param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=max_iters,
        by_epoch=False)
]

cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
cfg.default_hooks.visualization = dict(type='SegVisualizationHook', draw=True)

# https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py
cfg.default_hooks.checkpoint = dict(type='CheckpointHook', by_epoch=True, save_best=['MMSeg/mAcc', "OCELOT/mF1"], rule='greater')
cfg.visualizer  = dict(type='SegLocalVisualizer', vis_backends=[dict(type='LocalVisBackend'),
                                                     dict(type='WandbVisBackend')], name='visualizer')

cfg.val_evaluator = [
   dict(type='IoUMetric', iou_metrics=['mIoU'], prefix="MMSeg"), 
   dict(type='F1_OCELOTSegmentationScoreMetric', csv_annotation_folder=csv_annotations_gt, metadata_path=metadata_path)
   ]
cfg.test_evaluator = cfg.val_evaluator

cfg.dump(f"{experiment_name}_config.py")

# wandb.init(project="MMSegOZELOT", name=experiment_name, config=cfg, entity="christianml", save_code=True, mode="online") #

runner = Runner.from_cfg(cfg)
# start training
runner.train()