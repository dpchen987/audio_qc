#!/usr/bin/env python3
# coding:utf-8

import os
import json
import time
import asyncio
import argparse
import aiohttp
import statistics
import soundfile as sf
import base64

apis = [
    'http://127.0.0.1:8300/asr/v1/speech_vad',
    # 'http://127.0.0.1:8301/asr/v1/speech_vad',
    # 'http://127.0.0.1:8302/asr/v1/speech_vad',
    # 'http://127.0.0.1:8303/asr/v1/speech_vad',
]
api_idx = 0

audio_pth = './1680840021629.opus'
with open(audio_pth, 'rb') as fin:
        audio_data = fin.read()
bs64 = base64.b64encode(audio_data).decode('latin1')


def print_result(info):
    length = max([len(k) for k in info])
    for k, v in info.items():
        print(f'\t{k: >{length}} : {v}')


async def test_coro(task_id, times):
    global api_idx
    api = apis[api_idx]
    api_idx += 1
    api_idx %= len(apis)
    print(api)
    data = {
        "task_id": str(task_id),
        "file_path": "/data/audio/speech.pcm",
        "callback_url": "http://localhost:8305/asr/v1/callBack_test",
        "file_content": bs64,
        "file_type": "opus",
        "trans_type": 2
    }
    begin = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(api, json=data) as resp:
            _ = await resp.text()
            # print(_)
    time_cost = time.time() - begin
    times.append(time_cost)


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-t', '--total', type=int, default=100,
        help='total queries for this run')
    parser.add_argument(
        '-n', '--num_concurrence', type=int, required=True,
        help='num of concurrence for query')
    args = parser.parse_args()
    return args

async def main(args):
    tasks = set()
    request_times = []
    print('starting...')
    begin = time.time()
    for i in range(args.total):
        task = asyncio.create_task(test_coro(i, request_times))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        if len(tasks) < args.num_concurrence:
            continue
        print((f'{i=}, start {args.num_concurrence} '
               f'queries @ {time.strftime("%m-%d %H:%M:%S")}'))
        await asyncio.sleep(0.1)
        while len(tasks) >= args.num_concurrence:
            print('waitting...')
            await asyncio.sleep(0.3)
    while tasks:
        print('waitting...')
        await asyncio.sleep(0.8)
    request_time = time.time() - begin
    concur_info = {
        'request_time': request_time,
    }
    request_info = {
        'mean': statistics.mean(request_times),
        'median': statistics.median(request_times),
        'max_time': max(request_times),
        'min_time': min(request_times),
    }
    print('For all concurrence:')
    print_result(concur_info)
    print('For one request:')
    print_result(request_info)
    # # caculate CER
    # cmd = (f'../test-model/compute-wer.py --char=1 --v=1 '
    #        f'--ig={args.ignore} '
    #        f'{args.trans} {args.save_to} > '
    #        f'{args.save_to}-test-{args.num_concurrence}.cer.txt')
    # print(cmd)
    # os.system(cmd)
    print('done')


if __name__ == '__main__':
    args = get_args()
    asyncio.run(main(args))
