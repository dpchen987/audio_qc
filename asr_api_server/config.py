# -*- coding: utf-8 -*-
'''
server的相关配置均由环境变量赋值，省去配置文件，方便部署
'''

import os
import re



CONF = dict(
    host='0.0.0.0',
    port=8300,
    ws=['ws://127.0.0.1:8301'],
)


def parse_env():
    global CONF
    CONF['host'] = os.getenv('SDC_HOST', CONF['host'])
    CONF['port'] = int(os.getenv('SDC_PORT', CONF['port']))
    ws = os.getenv('SDC_WS', '')
    if ws:
        ws = re.split(r'[,\s]+', ws)
        for w in ws:
            if not re.search(r':\d+'):
                raise ValueError(f'invalid WS address {w}')
        CONF['ws'] = ws


parse_env()
WS_INDEX = 0


def get_ws():
    global WS_INDEX
    if len(CONF['ws']) == 1:
        return CONF['ws'][0]
    idx = WS_INDEX / len(CONF['ws'])
    WS_INDEX = idx + 1
    return CONF['ws'][idx]