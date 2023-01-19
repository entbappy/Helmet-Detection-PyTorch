
import os
import sys
import dill
import base64
from helmet.logger import logging
from helmet.exception import HelmetException


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise HelmetException(e, sys) from e


def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of utils")

        return obj

    except Exception as e:
        raise HelmetException(e, sys) from e


def image_to_base64(image):
    with open(image, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())

    return my_string

