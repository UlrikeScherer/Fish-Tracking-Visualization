import sys
import logging
import logging.handlers


# TODO: logger
'''
* create logger w/name
* create logging streams with agnostic log level
* format logger
* create date-dependent logger
'''
def create_logger(
    logger_name: str,
    log_level_stream: int,
    log_level_file: int,
    filename: str,
):
    ''' 
    configuration of the logging verbosity
    params: 
        filename: consists of program execution mode and timestamp
        filemode: for opening the file in a specific mode (`a` for appending)
        level: either an integer value or directly a logger enum (`ERROR`==`40`, `WARNING`==30)
    '''

    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(name)s |  %(levelname)s: %(message)s')
    logger.setLevel(logging.DEBUG)
    
    log_stream_handler = create_log_stream_handler(
        log_level_stream,
        formatter
    )

    log_file_handler = create_log_file_handler(
        filename,
        log_level_file,
        formatter
    )

    logger.addHandler(log_stream_handler)
    logger.addHandler(log_file_handler)
    
    # count logging-level-agnostic calls
    logger.debug = CallCounted(logger.debug)
    logger.info = CallCounted(logger.info)
    logger.warning = CallCounted(logger.warning)
    logger.error = CallCounted(logger.error)
    logger.critical = CallCounted(logger.critical)
    return logger

def create_log_stream_handler(
    log_level_stream: int,
    formatter
):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level_stream)
    stream_handler.setFormatter(formatter)
    return stream_handler

def create_log_file_handler(
    filename: str,
    log_level_file: int,
    formatter
):
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=filename, when='midnight', backupCount=30)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level_file)
    return file_handler

class CallCounted:
    """Decorator to determine number of calls for a method"""
    def __init__(self,method):
        self.method=method
        self.counter=0

    def __call__(self,*args,**kwargs):
        self.counter+=1
        return self.method(*args,**kwargs)
