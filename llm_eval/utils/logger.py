import logging
import inspect
import os
from datetime import datetime


def init_global_log(level: str = "info", enable_save: bool = False, log_dir: str = "logs"):
    # Clear existing handlers if any
    logging.root.handlers = []
    
    log_level = getattr(logging, level.upper(), logging.INFO)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = '%(asctime)s [%(levelname)s] - %(message)s - %(filename)s:%(lineno)d'
    
    handlers = [logging.StreamHandler()]

    if enable_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"runtime_{timestamp}.log")
        handlers.append(logging.FileHandler(log_file_path))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing handlers
    )
    
    # Note: filter too many kafka logs
    kafka_logger = logging.getLogger("kafka")
    kafka_logger.setLevel(logging.ERROR)


class Logger:
    def __init__(self, level=None, name=__name__):
        self.logger = logging.getLogger(name)
        
        if level:
            self.logger.setLevel(level)
        else:        
            self.logger.setLevel(logging.INFO)
        
        if self.logger.level == logging.DEBUG:
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s - %(caller_filename)s:%(caller_lineno)d')
        else:
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
            
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def set_level(self, level):
        self.logger.setLevel(level)

    def debug(self, message):
        self.logger.debug(message, extra=self._get_extra_info())

    def info(self, message):
        self.logger.info(message, extra=self._get_extra_info())

    def warning(self, message):
        self.logger.warning(message, extra=self._get_extra_info())

    def error(self, message):
        self.logger.error(message, extra=self._get_extra_info())

    def critical(self, message):
        self.logger.critical(message, extra=self._get_extra_info())

    def _get_extra_info(self):
        frame = inspect.currentframe().f_back.f_back
        caller_filename = inspect.getframeinfo(frame).filename
        caller_lineno = inspect.getframeinfo(frame).lineno
        return {'caller_filename': caller_filename, 'caller_lineno': caller_lineno}

