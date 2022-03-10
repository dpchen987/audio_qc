#!/usr/bin/env python3

import time
import requests
import asyncio
import aiohttp


api = 'http://127.0.0.1:8300/asr/v1/rec'


def test_one(audio_file, url=False):
    headers = {
        'appkey': '123',
        'format': 'pcm',
        #'Content-Type': 'application/octet-stream',
    }
    if url:
        headers['audio-url'] = 'https://yinshuhuiyuan-oss-10001.oss.jingan-hlw.inspurcloudoss.com/video/AHC0022101232683/908159455844241408/1636615795209.opus'
    if 'audio-url' in headers:
        b = time.time()
        r = requests.post(api, headers=headers)
        e = time.time()
    else:
        with open(audio_file, 'rb') as f:
            data = f.read()
        b = time.time()
        r = requests.post(api, data=data, headers=headers)
        e = time.time()
    print(r.text)
    print('time cost:', e-b)


async def test_coro(i, audio_data):
    headers = {
        'appkey': '123',
        'format': 'pcm',
    }
    print('start ', i)
    b = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(api, data=audio_data, headers=headers) as resp:
            text = await resp.text()
    e = time.time()
    print(f'coro-{i} time cost: {e-b}, {len(text)=}')
    return text


async def test_multi(audio_file, count=16):
    b = time.time()
    print('start @', b)
    with open(audio_file, 'rb') as f:
        data = f.read()
    tasks = []
    for i in range(count):
        t = asyncio.create_task(test_coro(i, data))
        tasks.append(t)
    for t in tasks:
        await t
    e = time.time()
    print('done @', e)
    print('time ', e - b)



if __name__ == '__main__':
    from sys import argv
    if len(argv) == 2:
        fn = argv[1]
        test_one(fn, True)
    elif len(argv) == 3:
        fn = argv[1]
        count = int(argv[2])
        asyncio.run(test_multi(fn, count))
    else:
        print('bad args')
