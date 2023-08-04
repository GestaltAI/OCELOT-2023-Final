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
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Index images for the OZELOT challenge')
parser.add_argument('--input_folder', type=str, 
                    default="/mnt/g/Datasets/ocelot2023_v0.1.2/annotations/train/tissue", 
                    help='The ground truth csv file')
parser.add_argument('--cat_tumor_id', type=int, help='The CAT id for the tumor annotation type')
parser.add_argument('--cat_background_id', type=int, help='The CAT id for the background annotation type')
parser.add_argument('--cat_unknown_id', type=int, help='The CAT id for the unknown annotation type')
parser.add_argument('--image_set_id', type=int, help='ID of the image set to index the images in')
parser.add_argument('--username', type=str, help='CAT user name')
parser.add_argument('--password', type=str, help='CAT user password')
parser.add_argument('--host', type=str, help='CAT user name')
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
    for gt_path in tqdm(sorted(list(folder.glob("*.png")), key=lambda path: int(path.stem))):
        result_annos = []
        mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        image = images[Path(gt_path).stem]


        for label, annotation_type in zip([2, 1, 255], [args.cat_tumor_id, args.cat_background_id, args.cat_unknown_id]):


            label_map = ((mask == label) * 255).astype(np.uint8)
            # Create a back border arround the image to make sure that the contours are closed
            label_map = cv2.copyMakeBorder(label_map, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

            contours, hierarchy = cv2.findContours(label_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            for cnt, _ in zip(contours, hierarchy[0]):
                vector = {}
                for idx, point in enumerate(cnt):
                    vector[f"x{idx+1}"] = int((point[0][0]))
                    vector[f"y{idx+1}"] = int((point[0][1]))


                annotation_bbox = Annotation(annotation_type=annotation_type, 
                                                vector=vector, 
                                                image=image.id)
                result_annos.append(annotation_bbox)
        annotations_api.create_annotation(body=result_annos)


