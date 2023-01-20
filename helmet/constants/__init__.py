import os
import torch
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data Ingestion Constants
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME = 'helmet-object-detection'
ZIP_FILE_NAME = 'data.zip'
ANNOTATIONS_COCO_JSON_FILE = '_annotations.coco.json'

INPUT_SIZE = 416
HORIZONTAL_FLIP = 0.3
VERTICAL_FLIP = 0.3
RANDOM_BRIGHTNESS_CONTRAST = 0.1
COLOR_JITTER = 0.1
BBOX_FORMAT = 'coco'

RAW_FILE_NAME = 'helmet'

# Data ingestion constants 
DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'
DATA_INGESTION_TRAIN_DIR = 'train'
DATA_INGESTION_TEST_DIR = 'test'
DATA_INGESTION_VALID_DIR = 'valid'



# AWS CONSTANTS
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "ap-south-1"