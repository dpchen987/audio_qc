# -*- coding: utf-8 -*-
'''
server的相关配置均由环境变量赋值，省去配置文件，方便部署
'''

import os
import re
import leveldb


CONF = dict(
    host='0.0.0.0',
    port=8300,
    ws=['ws://127.0.0.1:8301'],
    url_db='',  # must be set as environmente
)


def parse_env():
    global CONF
    CONF['url_db'] = os.getenv('ASR_API_URL_DB', '')
    if not CONF['url_db']:
        raise ValueError('environmente ASR_API_URL_DB must be set!!!')
    CONF['host'] = os.getenv('ASR_API_HOST', CONF['host'])
    CONF['port'] = int(os.getenv('ASR_API_PORT', CONF['port']))
    ws = os.getenv('ASR_WS', '')
    if ws:
        ws = re.split(r'[,\s]+', ws)
        for w in ws:
            if not re.search(r':\d+', w):
                raise ValueError(f'invalid WS address {w}')
        CONF['ws'] = ws
        print('***'*10)
        print('websocket_server:', CONF['ws'])
        print('***'*10)


parse_env()
WS_INDEX = 0


def get_ws():
    global WS_INDEX
    if len(CONF['ws']) == 1:
        return CONF['ws'][0]
    idx = WS_INDEX % len(CONF['ws'])
    WS_INDEX = idx + 1
    return CONF['ws'][idx]


url_db = leveldb.LevelDB(CONF['url_db'])

processing_set = set()
background_tasks = set()
