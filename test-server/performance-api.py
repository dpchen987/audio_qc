#!/usr/bin/env python
# coding:utf-8

import os
import json
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


async def main(wav_scp_file, trans_text_file, concurrence):
    asr_trans = f'{wav_scp_file}-asr.txt'
    results = {}  # {uttid: text,}
    bad = 0
    if False and os.path.exists(asr_trans):
        with open(asr_trans) as f:
            for l in f:
                zz = l.strip().split('\t')
                if len(zz) != 2:
                    # print('bad', l)
                    bad += 1
                    continue
                results[zz[0]] = zz[1]
    print(f'{len(results)=}, {bad=}')

    wav_scp = []
    total_duration = 0
    with open(wav_scp_file) as f:
        for l in f:
            zz = l.strip().split('\t')
            if zz[0] in results:
                continue
            with open(zz[1], 'rb') as f:
                data = f.read()
            duration = (len(data) - 44)/2/16000
            total_duration += duration
            wav_scp.append((zz[0], data))
    print(f'{len(wav_scp)=}, {total_duration= }')

    f_result = open(asr_trans, 'w')
    for uttid, text in results.items():
        f_result.write(f'{uttid}\t{text}\n')
    f_result.flush()
    tasks = []
    failed = 0
    b = time.time()
    for _uttid, data in wav_scp:
        if len(tasks) < concurrence:
            t = asyncio.create_task(test_coro(data))
            tasks.append((_uttid, t))
            continue
        texts = []
        # print('waiting tasks...', len(tasks), i, time.strftime('%m-%d %H:%M:%S'))
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
        # print('waiting tasks...', i)
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
    print(f'{total_duration=}, {failed = }, time cost: {e-b}')
    # caculate CER
    cmd = (f'python ../test-model/compute-wer.py --char=1 --v=1 '
           f'{trans_text_file} {asr_trans} > {__file__}-test-{concurrence}.cer.txt')
    print(cmd)
    os.system(cmd)
    print('done')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-w', '--wav_scp', required=True, help='wav_scp_file')
    parser.add_argument('-t', '--trans', required=True, help='transcription file')
    parser.add_argument('-n', '--num_concurrence', type=int, default=1, help='num of concurrence for query')
    args = parser.parse_args()
    print('\n# test-1: single query')
    # asyncio.run(main(wav_scp_file, trans_text_file, concurrence=1))

    print('\n# test-2: multiple concurrence')
    asyncio.run(main(args.wav_scp, args.trans, concurrence=args.num_concurrence))
