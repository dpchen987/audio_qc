# -*- coding: utf-8 -*-
'''
server的相关配置均由运行参数赋值，省去配置文件，方便部署
'''

import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--host', type=str, default='127.0.0.1', help="监听IP地址")
parser.add_argument('--port', type=int, default=8300)
parser.add_argument('--ws', type=str, default='127.0.0.1:8301', help="ASR websocket 地址, 多个以英文逗号分隔: ip1:port1,ip2:port2, ...")

ARGS = parser.parse_args()
ws = re.split(r'[,\s]+', ARGS.ws)
good = []
for w in ws:
    if not w.startswith('ws://'):
        w = 'ws://' + w
    good.append(w)
ARGS.ws = good
print(ARGS.ws)
