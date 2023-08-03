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


import argparse

parser = argparse.ArgumentParser(description='Index images for the OZELOT challenge')
parser.add_argument('--input_folder', type=str, 
                    default="'/mnt/g/Datasets/ocelot2023_v0.1.2/images/train/cell", 
                    help='Path to the input folder')
parser.add_argument('--image_set_id', type=int, help='ID of the image set to index the images in')
parser.add_argument('--username', type=str, default="exact", 
                    help='CAT user name')
parser.add_argument('--password', type=str, default="exact", 
                    help='CAT user password')
parser.add_argument('--host', type=str, default="http://127.0.0.1:8000", 
                    # "http://azvm-mlops-b8.westus2.cloudapp.azure.com"
                    help='CAT user name')
args = parser.parse_args()


if __name__ == "__main__":


    configuration = Configuration()

    configuration.username = args.username
    configuration.password = args.password
    configuration.host = args.host


    client = ApiClient(configuration)
    image_sets_api = ImageSetsApi(client)
    images_api = ImagesApi(client)

    folder = Path(args.input_folder)
    for image_path in tqdm(list(folder.glob("*.jpg"))):

        _ = images_api.index_image(name=image_path.name, filename=image_path.name, image_set=args.image_set_id)


