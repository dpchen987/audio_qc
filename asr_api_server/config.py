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
    url_db='./db',  # must be set as environmente
    concurrency=10,
    backend='wenet',  # 'triton' or 'wenet'
    url=['127.0.0.1:8001'],  # triton-GRPCInferenceService
    download_timeout=20,  # audio download timeout seconds
)


def parse_env():
    global CONF
    CONF['url_db'] = os.getenv('ASR_API_URL_DB', CONF['url_db'])
    if not CONF['url_db']:
        raise ValueError('environmente ASR_API_URL_DB must be set!!!')
    CONF['host'] = os.getenv('ASR_API_HOST', CONF['host'])
    CONF['port'] = int(os.getenv('ASR_API_PORT', CONF['port']))
    CONF['backend'] = os.getenv('ASR_API_BACKEND', CONF['backend'])
    CONF['download_timeout'] = os.getenv('DOWNLOAD_TIMEOUT', CONF['download_timeout'])
    ws = os.getenv('ASR_WS', '')
    url = os.getenv('ASR_URL', '')
    assert CONF['backend'] != 'wenet' or CONF[
        'backend'] != 'triton', f'invalid backend, input `triton` or `wenet`, your value {CONF["backend"]}'
    print('***' * 10)
    print('ASR_API_BACKEND:', CONF['backend'])
    print('***' * 10)
    if ws:
        ws = re.split(r'[,\s]+', ws)
        for w in ws:
            if not re.search(r':\d+', w):
                raise ValueError(f'invalid WS address {w}')
        CONF['ws'] = ws
        print('***' * 10)
        print('websocket_server:', CONF['ws'])
        print('***' * 10)
    if url:
        url = re.split(r'[,\s]+', url)
        for u in url:
            if not re.search(r':\d+', u):
                raise ValueError(f'invalid ASR_URL address {u}')
        CONF['url'] = url
        print('***' * 10)
        print('triton server url:', CONF['url'])
        print('***' * 10)

    concur = os.getenv('ASR_API_CONCURRENCY', int(os.cpu_count() / 2))
    CONF['concurrency'] = int(concur)


parse_env()
WS_INDEX = 0
URL_INDEX = 0


def get_ws():
    global WS_INDEX
    if len(CONF['ws']) == 1:
        return CONF['ws'][0]
    idx = WS_INDEX % len(CONF['ws'])
    WS_INDEX = idx + 1
    return CONF['ws'][idx]


def get_url():
    global URL_INDEX
    if len(CONF['url']) == 1:
        return CONF['url'][0]
    idx = URL_INDEX % len(CONF['url'])
    URL_INDEX = idx + 1
    return CONF['url'][idx]


url_db = leveldb.LevelDB(CONF['url_db'])

processing_set = set()
background_tasks = set()
