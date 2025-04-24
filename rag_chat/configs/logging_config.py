import logging
# import coloredlogs

def setup_logging(level='INFO'):
    """Set up basic colored logging."""
    logger = logging.getLogger()
    # coloredlogs.install(level=level, logger=logger, fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s')