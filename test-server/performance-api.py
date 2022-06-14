#!/usr/bin/env python
# coding:utf-8

import os
import json
import time
import asyncio
import soundfile as sf
import aiohttp

api = 'http://127.0.0.1:8300/asr/v1/rec'


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


async def main(wav_scp_file, trans_text_file, concurrence):
    asr_trans = f'{wav_scp_file}-asr.txt'
    results = {}  # {uttid: text,}
    bad = 0
    if os.path.exists(asr_trans):
        with open(asr_trans) as f:
            for l in f:
                zz = l.strip().split('\t')
                if len(zz) != 2:
                    print('bad', l)
                    bad += 1
                    continue
                results[zz[0]] = zz[1]
    print(f'{len(results)=}, {bad=}')

    wav_scp = []
    with open(wav_scp_file) as f:
        for l in f:
            zz = l.strip().split('\t')
            wav_scp.append(zz)
    print(f'{len(wav_scp)=}')

    f_result = open(asr_trans, 'w')
    for uttid, text in results.items():
        f_result.write(f'{uttid}\t{text}\n')
    f_result.flush()
    tasks = []
    failed = 0
    b = time.time()
    for i, wav in enumerate(wav_scp):
        if wav[0] in results:
            # print('done', wav[0], i)
            continue
        if len(tasks) < concurrence:
            print('read', wav[1])
            with open(wav[1], 'rb') as f:
                data = f.read()
            print(f'{wav[1]}, {len(data)=}')
            t = asyncio.create_task(test_coro(i, data))
            tasks.append((wav[0], t))
            continue
        texts = []
        print('waiting tasks...', len(tasks), i, time.strftime('%m-%d %H:%M:%S'))
        for uttid, task in tasks:
            text = await task
            result = json.loads(text)
            if result['status'] != 2000:
                print('failed', uttid, result['message'])
                failed += 1
            texts.append(f'{uttid}\t{result["text"]}\n')
        f_result.write(''.join(texts))
        f_result.flush()
        tasks = []
    if tasks:
        print('waiting tasks...', i)
        texts = []
        for uttid, task in tasks:
            text = await task
            result = json.loads(text)
            if result['status'] != 2000:
                print('failed', uttid, result['message'])
                failed += 1
            texts.append(f'{uttid}\t{result["text"]}\n')
        f_result.write(''.join(texts))
        f_result.flush()
    f_result.close()
    e = time.time()
    print('time cost:', e-b)
    # caculate CER
    cmd = (f'python ../test-model/compute-wer.py --char=1 --v=1 '
           f'{trans_text_file} {asr_trans} > {__file__}-test-{concurrence}.cer.txt')
    print(cmd)
    os.system(cmd)
    print('done')


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print(f'Usage {sys.argv[0]} wav_scp_file trans_text_file')
        sys.exit(1)
    wav_scp_file = sys.argv[1]
    trans_text_file = sys.argv[2]

    print('\n# test-1: single query')
    asyncio.run(main(wav_scp_file, trans_text_file, concurrence=1))

    print('\n# test-2: multiple concurrence')
    asyncio.run(main(wav_scp_file, trans_text_file, concurrence=8))
