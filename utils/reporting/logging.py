import logging
from colorlog import ColoredFormatter

LOG_LEVEL = logging.DEBUG
LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
logger = logging.getLogger('pythonConfig')
logger.setLevel(LOG_LEVEL)
logger.addHandler(stream)
logger.propagate = False

def log_message(message, log_level=logging.DEBUG):
    message = str(message)
    print()
    if log_level==logging.DEBUG:
        logger.debug(message)
    elif log_level==logging.CRITICAL:
        logger.critical(message)
    elif log_level==logging.WARN:
        logger.warn(message)
    elif log_level==logging.ERROR:
        logger.error(message)
    else:
        logger.info(message)

