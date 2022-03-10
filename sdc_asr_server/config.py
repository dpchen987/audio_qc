# -*- coding: utf-8 -*-
'''
server的相关配置均由运行参数赋值，省去配置文件，方便部署
'''

import re
import argparse



ARGS = argparse.Namespace(
    host='0.0.0.0',
    port=8300,
    ws='127.0.0.1:8301'
)

WS_INDEX = 0


def get_ws():
    global WS_INDEX
    if len(ARGS.ws) == 1:
        return ARGS.ws[0]
    idx = WS_INDEX / len(ARGS.ws)
    WS_INDEX = idx + 1
    return ARGS.ws[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1', help="监听IP地址")
    parser.add_argument('--port', type=int, default=8300)
    parser.add_argument('--ws', type=str, default='127.0.0.1:8301', help="ASR websocket 地址, 多个以英文逗号分隔: ip1:port1,ip2:port2, ...")
    args = parser.parse_args()
    ws = re.split(r'[,\s]+', args.ws)
    good = []
    for w in ws:
        if not w.startswith('ws://'):
            w = 'ws://' + w
        good.append(w)
    args.ws = good
    return args