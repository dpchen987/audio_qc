#!/usr/bin/env python3
'''
test files in dir, save result to .txt
'''

import json
import os
import time
import asyncio
import aiohttp


api = 'http://127.0.0.1:8300/asr/v1/rec'


async def test_coro(audio_data):
    headers = {
        'appkey': '123',
        'format': 'pcm',
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(api, data=audio_data, headers=headers) as resp:
            text = await resp.text()
    return text


async def main(path_scp, path_trans):
    wavs = []
    with open(path_scp) as f:
        for l in f:
            zz = l.strip().split()
            if len(zz) != 2:
                print('invalid line:', l)
                continue
            key, audio_path = zz
            wavs.append((key, audio_path))
    print('wavs:', len(wavs))
    batch_size = 10
    batches = [wavs[i:i+batch_size] for i in range(0, len(wavs), batch_size)]
    print(f'{len(batches)=}')
    trans = open(path_trans, 'w+')
    b = time.time()
    print('start @', b)
    for batch in batches:
        tasks = []
        for key, wav in batch:
            with open(wav, 'rb') as f:
                audio_data = f.read()
            t = asyncio.create_task(test_coro(audio_data))
            tasks.append((key, wav, t))
        for key, wav, t in tasks:
            response = await t
            jsn = json.loads(response)
            text = jsn['result']
            trans.write(f'{key}\t{text}\n')
            trans.flush()
    e = time.time()
    trans.close()
    print('done @', e)
    print('time ', e - b)



if __name__ == '__main__':
    from sys import argv
    if len(argv) != 3:
        sys.stderr.write(f"{argv[0]} <in_scp> <out_trans>\n")
        exit(-1)

    path_scp = argv[1]
    path_trans = argv[2]
    asyncio.run(main(path_scp, path_trans))
