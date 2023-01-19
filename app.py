from helmet.logger import logging
from helmet.exception import HelmetException
import sys

logging.info("Welcome to this project")

try:
    a = 2 + '3'
    print(a)
except Exception as e:
    logging.info(HelmetException(e, sys))

    raise HelmetException(e, sys) from e   