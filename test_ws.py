#!/usr/bin/env python
# coding:utf-8


import time
import sys, os
import asyncio
path1 = os.path.dirname(__file__)
print(path1)
sys.path.append(path1)
# print(sys.path)

from asr_api_server.ws_query import ws_rec


pth = r'C:\Users\YJ-XXB-new1\Desktop\fig\7.wav'
async def main():
    print('start...')
    fn = pth
    with open(fn, 'rb') as f:
        data = f.read()
    b = time.time()
    text = await ws_rec(data)
    print(text)
    print(f'{len(text) = }', time.time() - b)


asyncio.run(main())
