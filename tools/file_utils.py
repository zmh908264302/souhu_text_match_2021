import os

from tools import logging

logger = logging.get_logger(__name__)

try:
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
    if USE_TORCH in ("1", "ON", "YES", "AUTO"):
        import torch

        _torch_available = True  # pylint: disable=invalid-name
        logger.info("PyTorch version {} available.".format(torch.__version__))
except ImportError:
    _torch_available = False  # pylint: disable=invalid-name
