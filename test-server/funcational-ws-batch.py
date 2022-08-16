#!/usr/bin/env python
# coding:utf-8


import time
import asyncio
import sys
import soundfile as sf
sys.path.append('../')

from asr_api_server.ws_query_batch import ws_rec


async def main(fn):
    print('start...')
    data, sr = sf.read(fn, dtype='int16')
    data = data.tobytes()
    b = time.time()
    text = await ws_rec([data, data])
    print(f'{len(text) = }', time.time() - b)
    for i, t in enumerate(text):
        print(i, t)


fn = sys.argv[1]
asyncio.run(main(fn))
