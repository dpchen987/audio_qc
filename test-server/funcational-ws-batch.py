#!/usr/bin/env python
# coding:utf-8


import time
import asyncio
import sys
import soundfile as sf
sys.path.append('../')

from performance_ws import ws_rec_batch

ws_uri = 'ws://127.0.0.1:8303'


async def main(fn, batch_size):
    print('start...')
    data, sr = sf.read(fn, dtype='int16')
    data = data.tobytes()
    b = time.time()
    text = await ws_rec_batch([data]*batch_size, ws_uri)
    text = text['texts']
    print(f'{len(text) = }, time cost ', time.time() - b)
    for i, t in enumerate(text):
        print(i, t)


async def process(files):
    print(files)
    wavs = []
    for f in files:
        data, sr = sf.read(f, dtype='int16')
        wavs.append(data.tobytes())
    b = time.time()
    text = await ws_rec_batch(wavs, ws_uri)
    text = text['texts']
    print(f'{len(text) = }', time.time() - b)
    for i, t in enumerate(text):
        print(i, t)


if __name__ == '__main__':
    fn = sys.argv[1]
    if fn == 'fs':
        files = sys.argv[2:]
        asyncio.run(process(files))
    else:
        batch_size = int(sys.argv[2])
        asyncio.run(main(fn, batch_size))
