from pathlib import Path
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmengine.config import Config
import mmcv
import matplotlib.pyplot as plt

from tqdm import tqdm

check_point_folder = Path("ocelot23algo/user/seg_yolo/checkpoints/segformer_b2")
config_path = 'config.py'
checkpoint_path = "iter_20000.pth"


cfg = Config.fromfile(check_point_folder / config_path)

model = init_model(cfg, str(check_point_folder / checkpoint_path), 'cuda:0')


img_folder = "/mnt/g/Datasets/ocelot2023_v0.1.2/images/train/tissue/"

outfolder = Path("temp")
outfolder.mkdir(exist_ok=True, parents=True)

for img_path in tqdm(list(Path(img_folder).glob("*.jpg"))):
    img = mmcv.imread(img_path)
    result = inference_model(model, img)
    #plt.figure(figsize=(8, 6))
    show_result_pyplot(model, img, result, show=False, out_file=str(outfolder / img_path.name), opacity=0.25)
    # vis_result = show_result_pyplot(model, img, resul)
    #plt.imshow(mmcv.bgr2rgb(vis_result))

    # result.pred_sem_seg.data.cpu().numpy()