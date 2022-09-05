#!/usr/bin/env python3
# coding:utf-8

# Copyright (c) 2022 SDCI Co. Ltd (author: veelion)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
import asyncio
import argparse
import aiohttp 
import soundfile as sf
import statistics

WORKERS = 0


def get_args():
    description = 'speed test of websocket_server_main with run_batch'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-u', '--uri', required=True,
        default='http://127.0.0.1:8300/asr/v1/rec',
        help="asr_api_server's uri")
    parser.add_argument(
        '-w', '--wav_path', required=True,
        help='path to wav')
    parser.add_argument(
        '-c', '--concurrency', type=int, required=True,
        help='number of requests to make at a time')
    parser.add_argument(
        '-n', '--requests', type=int, required=True,
        help='number of requests to perform')
    args = parser.parse_args()
    return args


def print_result(info):
    length = max([len(k) for k in info])
    for k, v in info.items():
        if isinstance(v, float):
            v = round(v, 4)
        print(f'\t{k: >{length}} : {v}')


async def worker(data, api, results, failures):
    headers = {
        'appkey': '123',
        'format': 'pcm',
    }
    global WORKERS
    WORKERS += 1
    print('start ', WORKERS)
    begin = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(api, data=data, headers=headers) as resp:
            text = await resp.text()
    time_cost = time.time() - begin
    jn = json.loads(text)
    if jn['status'] != 2000:
        failures.append(1)
    results.append(time_cost)
    print('done ', WORKERS)
    WORKERS -= 1


async def main():
    global WORKERS
    args = get_args()
    data, sr = sf.read(args.wav_path, dtype='int16')
    assert sr == 16000
    duration = (len(data)) / 16000
    total_duration = duration * args.requests
    print(f'{duration = }')
    with open(args.wav_path, 'rb') as f:
        data = f.read()
    tasks = []
    request_times = []
    failures = []
    begin = time.time()
    print(f'{args.requests = }, {args.concurrency = }')
    for _ in range(args.requests):
        task = asyncio.create_task(
                worker(data, args.uri, request_times, failures))
        tasks.append(task)
        if len(tasks) >= args.concurrency:
            t = tasks.pop(0)
            await t
    print('WORKERS === ', WORKERS)
    while tasks:
        t = tasks.pop(0)
        await t
    print('WORKERS === ', WORKERS)
    request_time = time.time() - begin
    rtf = request_time / total_duration
    print('For all concurrence:')
    print_result({
        'failures': len(failures),
        'total_duration': total_duration,
        'request_time': request_time,
        'RTF': rtf,
    })
    print('For one request:')
    print_result({
        'mean': statistics.mean(request_times),
        'median': statistics.median(request_times),
        'max_time': max(request_times),
        'min_time': min(request_times),
    })
    print('done')


if __name__ == '__main__':
    asyncio.run(main())
