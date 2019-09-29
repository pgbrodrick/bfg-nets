import logging
import logging.handlers
import sys

MAX_BYTES = int(100 * 1024 * 1024)  # 10 MB


def get_root_logger(log_outfile: str = None) -> logging.Logger:
    """Get a root logger configured to log to stdout and optionally to a log file. Note that bfgn log messages will
    also appear in the log files, so this is probably useful even if you don't write your own logging messages.

    Args:
        log_outfile: File to which logs should be written.

    Returns:
        Logger configured for logging.

    Example:
        >>> logger = get_root_logger('logfile.out')
        >>> logger.setLevel('INFO')
        >>> logger.info('This log message will be in logfile.out')
        >>> logger.debug('This log message will not appear because the log level is set to info.')
    """
    logger = logging.getLogger("bfgn")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s")
    # Stream handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # File handler
    if log_outfile is not None:
        handler = logging.handlers.RotatingFileHandler(log_outfile, maxBytes=MAX_BYTES)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
