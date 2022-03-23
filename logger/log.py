from loguru import logger
from datetime import datetime
from globals.globals import LOG_FILE_PATH,LOG_LEVEL
import sys
# Remove current handlers
logger.remove()
logger.add(f"{LOG_FILE_PATH}")
logger.add(sys.stderr, level = LOG_LEVEL)

logger = logger


