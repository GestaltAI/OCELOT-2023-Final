# Description


We trained a SegFormer-B2 with three classes:
- Background 0
- Tumor 1

To reproduce the results download the model from the following [URL](https://drive.google.com/drive/folders/167hCNIdbBlFF8HkZ-gDqS8KpnOVBybNl?usp=sharing) and copy the files to checkpoints. 

Afterwards, in the process.py import 
```
from user.seg_unet_2class.model import Model
```

and run it to create a cell_classification.json file with the following configuration.

```json
        {
            "name": "CM: RUN PROCESS",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/ocelot23algo/process.py",
            "cwd": "${workspaceFolder}/ocelot23algo",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
            ],
            "env": {
                "GC_CELL_FPATH":            "${workspaceFolder}/ocelot23algo/test/fold_0/input/images/cell_patches/",
                "GC_TISSUE_FPATH":          "${workspaceFolder}/ocelot23algo/test/fold_0/input/images/tissue_patches/",
                "GC_METADATA_FPATH":        "${workspaceFolder}/ocelot23algo/test/fold_0/input/metadata.json",
                "GC_DETECTION_OUTPUT_PATH": "${workspaceFolder}/ocelot23algo/test/fold_0/output/cell_classification.json",
            }
        },
```

Afterwards, run the evaluation script.

```json
        {
            "name": "CM: EVAL PROCESS",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/ocelot23algo/evaluation/eval.py",
            "cwd": "${workspaceFolder}/ocelot23algo",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                "--gt_file",
                "/mnt/c/ProgProjekte/Python/OCELOTMICCAI23/ocelot23algo/test/fold_0/output/gt.json",
                "--prediction_file",
                "/mnt/c/ProgProjekte/Python/OCELOTMICCAI23/ocelot23algo/test/fold_0/output/cell_classification.json"
            ]
        }
```