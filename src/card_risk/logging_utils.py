import logging


def get_logger(name: str) -> logging.Logger:
    '''
    Returns a logger with a consistent format for all pipeline stages.
    :param name: logger name, usually __name__
    :return: logging.Logger
    '''
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
