#!/usr/bin/env python
# coding:utf-8


import os
import asyncio



async def worker(url_audio, url_callback):
    '''
    1. call websocket_server
    2. send result to url_callback
    3. delete url_audio from db
    '''

async def consume():
    concurrency = os.cpu_count // 2
    print(f'set {concurrency = }')
    while 1:
        print('consuming ...')
        # for _ in range(concurrency):
        #     url_audio, url_callback = get_from_db()
        #     asyncio.creat_task(worker(url_audio, url_callback))
        await asyncio.sleep(1)
