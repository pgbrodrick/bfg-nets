import logging
import logging.handlers
import sys


MAX_BYTES = int(100 * 1024 * 1024)  # 10 MB


"""
How to use logging:

Throughout the modules in your packages:
```
from gdcs.utils import logging
_logger = logging.get_child_logger(__name__)
_logger.trace('This is a low level log message that is very detailed, only for the worst debugging')
_logger.debug('This is a medium level log message, useful when you're generally debugging')
_logger.info('This is a high level log message, useful when you're confirming or expect that things are working')
_logger.warning('This is a warning for when something is wrong but you don't want to halt the program')
_logger.error('This is an error for what something is very very bad and the program should be halted')
# Although it's probably better to do something like:
assert is_everything_okay is True, 'This message gets put into the log errors anyway'
```

In your main script that runs analyses:
```
from gdcs.utils import logging
_logger = logging.get_root_logger('model_directory/log.out')
```
Then, all of the child loggers will pass their messages to the root logger and all the messages will appear at
model_directory/log.out.
"""


def get_child_logger(logger_name: str) -> logging.Logger:
    _configure_logging()
    return logging.getLogger(logger_name)


def get_root_logger(log_outfile: str = None) -> logging.Logger:
    _configure_logging()
    logger = logging.getLogger('rsCNN')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
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


def _configure_logging():
    _add_logging_level('TRACE', logging.DEBUG - 5)


def _add_logging_level(level_name: str, level_num: int) -> None:
    method_name = level_name.lower()
    if hasattr(logging, level_name) or hasattr(logging, method_name) or hasattr(logging.getLoggerClass(), method_name):
        return

    def log_for_level(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def log_to_root(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, log_for_level)
    setattr(logging, method_name, log_to_root)
