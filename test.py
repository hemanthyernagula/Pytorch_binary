# from logger.log import logger
from tqdm import trange
import time
# logger.debug("Testing")
# logger.success("Success")
# logger.warning("Warning")
# logger.error("Error")
# logger.info("Info")
# logger.critical("Critical")


progress = trange(10,desc='Epochs',leave=True)


# for i in progress:
#     for j in range(5):
#         progress.set_description(f"Epochs {i} Batch {j}")
#         time.sleep(1)
#     time.sleep(1)


