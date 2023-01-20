import os
import pickle
import sys
from io import StringIO
from typing import List, Union
from botocore.exceptions import ClientError
import boto3
from helmet.exception import HelmetException
from helmet.logger import logging
from mypy_boto3_s3.service_resource import Bucket
from helmet.constants import *

MODEL_SAVE_FORMAT = ".pt"


class S3Operation:
    s3_client=None
    s3_resource = None
    def __init__(self):
        # self.s3_client = boto3.client("s3")

        # self.s3_resource = boto3.resource("s3")
        if S3Operation.s3_resource==None or S3Operation.s3_client==None:
            __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY, )
            __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY, )
            if __access_key_id is None:
                raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not set.")
            if __secret_access_key is None:
                raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set.")
        
            S3Operation.s3_resource = boto3.resource('s3',
                                            aws_access_key_id=__access_key_id,
                                            aws_secret_access_key=__secret_access_key,
                                            region_name=REGION_NAME
                                            )
            S3Operation.s3_client = boto3.client('s3',
                                        aws_access_key_id=__access_key_id,
                                        aws_secret_access_key=__secret_access_key,
                                        region_name=REGION_NAME
                                        )
        self.s3_resource = S3Operation.s3_resource
        self.s3_client = S3Operation.s3_client
    @staticmethod
    def read_object(
        object_name: str, decode: bool = True, make_readable: bool = False
    ) -> Union[StringIO, str]:
        """
        Method Name :   read_object
        Description :   This method reads the object_name object with kwargs

        Output      :   The column name is renamed
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the read_object method of S3Operations class")

        try:
            func = (
                lambda: object_name.get()["Body"].read().decode()
                if decode is True
                else object_name.get()["Body"].read()
            )
            conv_func = lambda: StringIO(func()) if make_readable is True else func()

            logging.info("Exited the read_object method of S3Operations class")

            return conv_func()

        except Exception as e:
            raise HelmetException(e, sys) from e

    def get_bucket(self, bucket_name: str) -> Bucket:
        """
        Method Name :   get_bucket
        Description :   This method gets the bucket object based on the bucket_name

        Output      :   Bucket object is returned based on the bucket name
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the get_bucket method of S3Operations class")

        try:
            bucket = self.s3_resource.Bucket(bucket_name)

            logging.info("Exited the get_bucket method of S3Operations class")

            return bucket

        except Exception as e:
            raise HelmetException(e, sys) from e

    def get_file_object(
        self, filename: str, bucket_name: str
    ) -> Union[List[object], object]:
        """
        Method Name :   get_file_object
        Description :   This method gets the file object from bucket_name bucket based on filename

        Output      :   list of objects or object is returned based on filename
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the get_file_object method of S3Operations class")

        try:
            bucket = self.get_bucket(bucket_name)

            lst_objs = [object for object in bucket.objects.filter(Prefix=filename)]

            func = lambda x: x[0] if len(x) == 1 else x

            file_objs = func(lst_objs)

            logging.info("Exited the get_file_object method of S3Operations class")

            return file_objs

        except Exception as e:
            raise HelmetException(e, sys) from e

    def load_model(
        self, model_name: str, bucket_name: str, model_dir: str = None
    ) -> object:
        """
        Method Name :   load_model
        Description :   This method loads the model_name model from bucket_name bucket with kwargs

        Output      :   list of objects or object is returned based on filename
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the load_model method of S3Operations class")

        try:
            func = (
                lambda: model_name
                if model_dir is None
                else model_dir + "/" + model_name
            )

            model_file = func()

            f_obj = self.get_file_object(model_file, bucket_name)

            model_obj = self.read_object(f_obj, decode=False)

            return model_obj
            logging.info("Exited the load_model method of S3Operations class")

        except Exception as e:
            raise HelmetException(e, sys) from e

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """
        Method Name :   create_folder
        Description :   This method creates a folder_name folder in bucket_name bucket

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the create_folder method of S3Operations class")

        try:
            self.s3_resource.Object(bucket_name, folder_name).load()

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                folder_obj = folder_name + "/"

                self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj)

            else:
                pass

            logging.info("Exited the create_folder method of S3Operations class")

    def upload_file(
        self,
        from_filename: str,
        to_filename: str,
        bucket_name: str,
        remove: bool = True,
    ):
        """
        Method Name :   upload_file
        Description :   This method uploads the from_filename file to bucket_name bucket with to_filename as bucket filename

        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the upload_file method of S3Operations class")

        try:
            logging.info(
                f"Uploading {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            self.s3_resource.meta.client.upload_file(
                from_filename, bucket_name, to_filename
            )

            logging.info(
                f"Uploaded {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            if remove is True:
                os.remove(from_filename)

                logging.info(f"Remove is set to {remove}, deleted the file")

            else:
                logging.info(f"Remove is set to {remove}, not deleted the file")

            logging.info("Exited the upload_file method of S3Operations class")

        except Exception as e:
            raise HelmetException(e, sys) from e

    
    def read_data_from_s3(self, filename: str, bucket_name: str, output_filename: str):
        try:
            bucket = self.get_bucket(bucket_name)
            
            obj = bucket.download_file(Key=filename, Filename=output_filename)

            return output_filename
            
        except Exception as e:
            raise HelmetException(e, sys) from e
