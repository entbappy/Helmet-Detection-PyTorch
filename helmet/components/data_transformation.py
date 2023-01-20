import os
import sys
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from helmet.logger import logging
from helmet.exception import HelmetException
from helmet.ml.feature.helmet_detection import HelmetDetection
from helmet.constants import *
from helmet.utils.main_utils import save_object
from helmet.entity.config_entity import DataTransformationConfig
from helmet.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact
    


    def number_of_classes(self):

        try:

            coco = COCO(os.path.join(self.data_ingestion_artifact.train_file_path, ANNOTATIONS_COCO_JSON_FILE))
            categories = coco.cats
            classes = [i[1]['name'] for i in categories.items()]
            n_classes = len(classes)

            return n_classes
        except Exception as e:
            raise HelmetException(e, sys) from e


    
    def get_transforms(self, train=False):
        try: 
            if train:
                transform = A.Compose([
                    A.Resize(INPUT_SIZE, INPUT_SIZE),
                    A.HorizontalFlip(p=HORIZONTAL_FLIP),
                    A.VerticalFlip(p=VERTICAL_FLIP),
                    A.RandomBrightnessContrast(p=RANDOM_BRIGHTNESS_CONTRAST),
                    A.ColorJitter(p=COLOR_JITTER),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format=BBOX_FORMAT))
            else:
                transform = A.Compose([
                    A.Resize(INPUT_SIZE, INPUT_SIZE), 
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format=BBOX_FORMAT))
            return transform
        except Exception as e:
            raise HelmetException(e, sys) from e


    
    def initiate_data_transformation(self) -> DataTransformationArtifacts:

        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")

            n_classes = self.number_of_classes()
            print(n_classes)

            logging.info(f"Total number of classes: {n_classes}")

            train_dataset = HelmetDetection(root=self.data_transformation_config.ROOT_DIR,
                                            split=self.data_transformation_config.TRAIN_SPLIT,
                                            transforms=self.get_transforms(True))

            logging.info(f"Training dataset prepared")

            test_dataset = HelmetDetection(root=self.data_transformation_config.ROOT_DIR,
                                           split=self.data_transformation_config.TEST_SPLIT,
                                           transforms=self.get_transforms(False))

            logging.info(f"Testing dataset prepared")

            save_object(self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH, train_dataset)
            save_object(self.data_transformation_config.TEST_TRANSFORM_OBJECT_FILE_PATH, test_dataset)

            logging.info("Saved the train transformed object")

            data_transformation_artifact = DataTransformationArtifacts(
                transformed_train_object=self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH,
                transformed_test_object=self.data_transformation_config.TEST_TRANSFORM_OBJECT_FILE_PATH,
                number_of_classes=n_classes)

            logging.info("Exited the initiate_data_transformation method of Data transformation class")

            return data_transformation_artifact

        except Exception as e:
            raise HelmetException(e, sys) from e



