#!/usr/bin/env python
# coding:utf-8


import time
import asyncio
import sys
sys.path.append('../')
import soundfile as sf

from performance_ws import ws_rec


async def main(fn):
    ws_uri = 'ws://127.0.0.1:8303'
    print('start...')
    # with open(fn, 'rb') as f:
    #     data = f.read()
    data, sr = sf.read(fn, dtype='int16')
    data = data.tobytes()
    b = time.time()
    text = await ws_rec(data, ws_uri)
    print(text)
    print(f'{len(text) = }', time.time() - b)


fn = sys.argv[1]
asyncio.run(main(fn))
