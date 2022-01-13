 # -*- coding: utf-8 -*-

'''
创建一个同时输出到屏幕和文件的logger
'''

import sys
import logging


def create_nofile():
    logger = logging.getLogger('aqc')
    fmt='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)#设置日志格式
    logger.setLevel(logging.DEBUG)#设置日志级别
    sh = logging.StreamHandler(sys.stdout)#往屏幕上输出
    sh.setFormatter(format_str) #设置屏幕上显示的格式
    logger.addHandler(sh) #把对象加到logger里
    return logger

 
logger = create_nofile()