"""
Данный модуль содержит реализацию логера
"""

import logging
import logging.config

class LoggerFormating(logging.Formatter):
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)'

    FORMATS = {logging.INFO: format}

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_file_handler():
    '''
    Функция для создания хандлера для вывода в файл

    Returns
    ---------
    `logging.FileHandler`
        Хандлер
    '''
    _log_format = "%(asctime)s\t%(levelname)s\t%(name)s\t" \
                  "%(filename)s.%(funcName)s " \
                  "line: %(lineno)d | \t%(message)s"
    file_handler = logging.FileHandler("cache.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_log_format))
    return file_handler


def get_stream_handler():
    '''
    Функция для создания хандлера для stdout

    Returns
    ---------
    `logging.StreamHandler`
        Хандлер
    '''
    _log_format = "%(asctime)s\t%(levelname)s\t %(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter(_log_format))

    return stream_handler


def get_logger(name, level, write_to_stdout=True, write_to_file=False):
    '''
    Функция для получения логера

    Parameters
    -----------
    name: `str`
        Имя логера
    level: `int`
        Уровень логирования
    write_to_stdout: `bool`
        True если записывать в stdout иначе False

    Returns
    -----------
    `logger`
        Логер
    '''
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if write_to_file:
        logger.addHandler(get_file_handler())
    if write_to_stdout:
        logger.addHandler(get_stream_handler())

    return logger