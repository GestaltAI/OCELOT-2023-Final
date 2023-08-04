from exact_sync.v1.api.annotations_api import AnnotationsApi
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api.image_sets_api import ImageSetsApi
from exact_sync.v1.api.teams_api import TeamsApi

from exact_sync.v1.models import ImageSet, Team, Product, AnnotationType, Image, Annotation, AnnotationMediaFile
from exact_sync.v1.rest import ApiException
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient

from tqdm import tqdm
from pathlib import Path

import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Index images for the OZELOT challenge')
parser.add_argument('--input_folder', type=str, 
                    default="/mnt/g/Datasets/ocelot2023_v0.1.2/annotations/train/cell", 
                    help='The ground truth csv file')
parser.add_argument('--cat_tumor_id', type=int, help='The CAT id for the tumor annotation type')
parser.add_argument('--cat_non_tumor_id', type=int, help='The CAT id for the non tumor annotation type')
parser.add_argument('--image_set_id', type=int, help='ID of the image set to index the images in')
parser.add_argument('--username', type=str, help='CAT user name')
parser.add_argument('--password', type=str, help='CAT user password')
parser.add_argument('--host', type=str,  help='CAT user name')
args = parser.parse_args()


if __name__ == "__main__":

    configuration = Configuration()

    configuration.username = args.username
    configuration.password = args.password
    configuration.host = args.host


    client = ApiClient(configuration)
    image_sets_api = ImageSetsApi(client)
    images_api = ImagesApi(client)
    annotations_api = AnnotationsApi(client)

    images = {Path(image.name).stem: image for image in images_api.list_images(image_set=args.image_set_id, pagination=False).results}

    folder = Path(args.input_folder)
    for gt_path in tqdm(sorted(list(folder.glob("*.csv")), key=lambda path: int(path.stem))):
        result_annos = []
        df = pd.read_csv(gt_path, names=["x", "y", "tumor"])

        image = images[Path(gt_path).stem]

        for _, row in df.iterrows():

            x, y = row["x"], row["y"]
            tumor = row["tumor"]

            annotation_type = args.cat_tumor_id if tumor == 2 else args.cat_non_tumor_id

            image = images[Path(gt_path).stem]

            vector_bbox = {
                "x1": max(0, int(x - 15)),
                "y1": max(0, int(y - 15)),

                #"x2": min(int(x + 15), image.width),
                #"y2": max(int(y - 15), 0),

                "x2": min(int(x + 15), image.width),
                "y2": min(int(y + 15), image.height),

                #"x4": max(int(x - 15), 0),
                #"y4": min(int(y + 15), image.height),
            }  

            annotation_bbox = Annotation(annotation_type=annotation_type, 
                                            vector=vector_bbox, 
                                            image=image.id)
            result_annos.append(annotation_bbox)

        annotations_api.create_annotation(body=result_annos)

