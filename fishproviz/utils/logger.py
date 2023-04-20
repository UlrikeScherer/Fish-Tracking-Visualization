import os.path
import logging
import logging.handlers

from fishproviz.utils.utile import get_timestamp, create_directory


def create_logger(
    logger_name: str,
    log_level_stream: int,
    log_level_file: int,
):
    ''' 
    configuration of the logging verbosity
    params: 
        logger_name: program name, overloaded to logging-instance, names the log-file
        log_level_stream: log level for stdout (e.g. `INFO`=`20`)
        log_level_file: log level for log-file (e.g.`DEBUG`=`10`)
    '''

    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(name)s |  %(levelname)s: %(message)s')
    logger.setLevel(logging.DEBUG)
    
    log_stream_handler = create_log_stream_handler(
        log_level_stream,
        formatter
    )
    
    logger.filepath = create_filepath_with_timestamp(logger_name)
    log_file_handler = create_log_file_handler(
        logger.filepath,
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


def create_filepath_with_timestamp(
    program_name: str,
):
    timestamp = get_timestamp()
    dir = create_directory('logs')
    filename = f'{program_name}_{timestamp}.log'
    filepath = os.path.join(
        dir,
        filename
    )
    return filepath


class CallCounted:
    """Decorator to determine number of calls for a method"""
    def __init__(self,method):
        self.method=method
        self.counter=0

    def __call__(self,*args,**kwargs):
        self.counter+=1
        return self.method(*args,**kwargs)
