import os
import sys
from zipfile import ZipFile
from helmet.entity.config_entity import DataIngestionConfig
from helmet.entity.artifacts_entity import DataIngestionArtifacts
from helmet.configuration.s3_operations import S3Operation
from helmet.exception import HelmetException
from helmet.logger import logging
from helmet.constants import *


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, s3_operations: S3Operation):
        self.data_ingestion_config = data_ingestion_config
        self.s3_operations = s3_operations

    
    def get_data_from_s3(self) -> None:
        try:
            logging.info("Entered the get_data_from_s3 method of Data ingestion class")
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            self.s3_operations.read_data_from_s3(self.data_ingestion_config.ZIP_FILE_NAME,
                                                 self.data_ingestion_config.BUCKET_NAME,
                                                 self.data_ingestion_config.ZIP_FILE_PATH)
            logging.info("Exited the get_data_from_s3 method of Data ingestion class")
        except Exception as e:
            raise HelmetException(e, sys) from e

    
    def unzip_and_clean(self):
        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try:
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)
            logging.info("Exited the unzip_and_clean method of Data ingestion class")

            return self.data_ingestion_config.TRAIN_DATA_ARTIFACT_DIR, self.data_ingestion_config.TEST_DATA_ARTIFACT_DIR, self.data_ingestion_config.VALID_DATA_ARTIFACT_DIR
        except Exception as e:
            raise HelmetException(e, sys) from e


    
    def initiate_data_ingestion(self) -> DataIngestionArtifacts: 
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")
        try:
            self.get_data_from_s3()

            logging.info("Fetched the data from S3 bucket")

            train_file_path, test_file_path, valid_file_path= self.unzip_and_clean()

            logging.info("Unzipped file and splited into train, test and valid")

            data_ingestion_artifact = DataIngestionArtifacts(train_file_path=train_file_path, 
                                                                test_file_path=test_file_path,
                                                                valid_file_path=valid_file_path)

            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise HelmetException(e, sys) from e