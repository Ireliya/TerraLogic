"""
RemoteSAM package for spatial reasoning agent.
"""

# Fix mmcv compatibility issues - add missing functions to mmcv if needed
try:
    from mmcv.utils import is_list_of
except ImportError:
    try:
        # In newer mmcv versions, is_list_of is in mmengine
        from mmengine.utils import is_list_of
        # Add it to mmcv.utils for backward compatibility
        import mmcv.utils
        mmcv.utils.is_list_of = is_list_of
    except ImportError:
        # Fallback implementation
        def is_list_of(obj, expected_type):
            """Check if obj is a list of expected_type."""
            if not isinstance(obj, list):
                return False
            return all(isinstance(item, expected_type) for item in obj)

        import mmcv.utils
        mmcv.utils.is_list_of = is_list_of

# Fix print_log import issue
try:
    from mmcv import print_log
except ImportError:
    try:
        # In newer mmcv versions, print_log might be in mmengine
        from mmengine.logging import print_log
        # Add it to mmcv for backward compatibility
        import mmcv
        mmcv.print_log = print_log
    except ImportError:
        # Fallback implementation using standard logging
        import logging
        def print_log(msg, logger=None, level=logging.INFO):
            """Fallback print_log implementation."""
            if logger is None:
                print(msg)
            else:
                if isinstance(logger, str):
                    logger = logging.getLogger(logger)
                logger.log(level, msg)

        import mmcv
        mmcv.print_log = print_log

# Import essential components
from . import utils
from . import transforms
from . import args
from . import RuleBasedCaptioning
from . import segmentation
from . import arc
from . import bert

# Import specific functions that might be needed
from .utils import M2B, EPOC
from .transforms import Compose, Resize, ToTensor, Normalize
from .args import get_parser, get_default_args
from .RuleBasedCaptioning import single_captioning

__all__ = [
    'utils',
    'transforms',
    'args',
    'RuleBasedCaptioning',
    'segmentation',
    'arc',
    'bert',
    'M2B',
    'EPOC',
    'Compose',
    'Resize',
    'ToTensor',
    'Normalize',
    'get_parser',
    'get_default_args',
    'single_captioning'
]
