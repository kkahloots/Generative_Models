import logging
from colorlog import ColoredFormatter

LOG_LEVEL = logging.DEBUG
LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
log = logging.getLogger('pythonConfig')
log.setLevel(LOG_LEVEL)
log.addHandler(stream)

def log_message(message, log_level=logging.DEBUG):
    message = str(message)
    print()
    if log_level==logging.DEBUG:
        log.debug(message)
    elif log_level==logging.CRITICAL:
        log.critical(message)
    elif log_level==logging.WARN:
        log.warn(message)
    elif log_level==logging.ERROR:
        log.error(message)
    else:
        log.info(message)

