#!/usr/bin/env python
# coding:utf-8


import time
import asyncio
import sys
sys.path.append('../')

from asr_api_server.ws_query import ws_rec


async def main(fn):
    print('start...')
    with open(fn, 'rb') as f:
        data = f.read()
    b = time.time()
    text = await ws_rec(data)
    print(text)
    print(f'{len(text) = }', time.time() - b)


fn = sys.argv[1]
asyncio.run(main(fn))
