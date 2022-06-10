#!/usr/bin/env python
# coding:utf-8


import time
import sys
import asyncio
from asr_api_server.ws_query import ws_rec


async def main():
    print('start...')
    fn = sys.argv[1]
    with open(fn, 'rb') as f:
        data = f.read()
    b = time.time()
    text = await ws_rec(data)
    print(text)
    print(f'{len(text) = }', time.time() - b)


asyncio.run(main())
