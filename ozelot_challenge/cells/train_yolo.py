import argparse
import wandb
from pathlib import Path
from ultralytics import YOLO


parser = argparse.ArgumentParser(description='Create the tissue dataset for MMSegmentation')
parser.add_argument('--input_file', type=str, 
                    default="/mnt/g/Datasets/ocelot2023_v0.1.2/MMSegCellsYolo/Folds/fold_0_local.yaml", 
                    help='Path to the ouput folder')
parser.add_argument('--model', type=str, 
                    default="s", 
                    help='Size: n,s,m,l,x')

args = parser.parse_args()


if __name__ == "__main__":
    # Load a model
    model = YOLO(f"yolov8{args.model}.yaml")  # build a new model from scratch
    model = YOLO(f"yolov8{args.model}.pt")  # load a pretrained model (recommended for training)



    folder = Path(args.input_file)


    # # Train the model
    # https://github.com/ultralytics/ultralytics/blob/3ae81ee9d11c432189e109d7a1724635a2e451ca/docs/usage/hyperparameter_tuning.md

    # https://learnopencv.com/slicing-aided-hyper-inference/
    # https://github.com/search?q=repo%3Aobss%2Fsahi%20yolov8&type=code
    
    results = model.train(data=args.input_file, 
                          
                        #entity="christianml",
                        project="YOLOV8_OCELOT_Cells",
                        name=f"yolov8{args.model}_fold_0_local",
                        
                        epochs=1, 
                        imgsz=960, 
                        batch=2,
                        #workers=0,
                        cache=False,
                        pretrained=True,
                        
                        #overlap_mask=False, # masks should overlap during training Not working!!!
                        
                        dropout=0.1,
                        half=True,

                        scale=0.1,
                        label_smoothing=0.1,
                        degrees=45,  # image rotation (+/- deg)
                        flipud=0.5,  # image flip up-down (probability)
                        mixup=0.1,  # image mixup (probability)
                        copy_paste=0.1,  # image copy-paste (probability)
                        mosaic=0.1,  # image mosaic (probability)
            )
    results_val = model.val()
    print(results_val)

    model.export()  