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

import os
import json
import time
import asyncio
import argparse
import websockets
import soundfile as sf
import statistics


WS_START = {
    'signal': 'start',
    'nbest': 1,
    'batch_lens': [],
    'enable_timestamp': False,
}

WS_END = json.dumps({
    'signal': 'end'
})

WORKERS = 0


async def ws_rec(data, ws_uri):
    assert isinstance(data, list)
    conn = await websockets.connect(ws_uri, ping_timeout=200)
    # async with websockets.connect(ws) as conn:
    # step 1: send start
    WS_START['batch_lens'] = [len(d) for d in data]
    await conn.send(json.dumps(WS_START))
    await conn.recv()
    # step 2: send audio data
    await conn.send(b''.join(data))
    result = await conn.recv()
    jn = json.loads(result)
    texts = []
    if jn['status'] != 'ok':
        print('failed from ws :', jn['message'])
    else:
        for result in jn['batch_result']:
            texts.append(result['nbest'][0]['sentence'])
    try:
        await conn.close()
    except Exception as e:
        # this except has no effect, just log as debug
        print(e)
    return texts


def get_args():
    description = 'speed test of websocket_server_main with run_batch'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-u', '--ws_uri', required=True,
        help="websocket_server_main's uri, e.g. ws://127.0.0.1:10086")
    parser.add_argument(
        '-w', '--wav_path', required=True,
        help='path to wav')
    parser.add_argument(
        '-c', '--concurrency', type=int, required=True,
        help='number of requests to make at a time')
    parser.add_argument(
        '-n', '--requests', type=int, required=True,
        help='number of requests to perform')
    parser.add_argument(
        '-b', '--batch_size', type=int, required=True,
        help='batch size of ws query')
    args = parser.parse_args()
    return args


def print_result(info):
    length = max([len(k) for k in info])
    for k, v in info.items():
        if isinstance(v, float):
            v = round(v, 4)
        print(f'\t{k: >{length}} : {v}')


async def worker(batch_data, ws_uri, results, failures):
    global WORKERS
    WORKERS += 1
    print('start ', WORKERS)
    b = time.time()
    text = await ws_rec(batch_data, ws_uri)
    if not text:
        failures.append(1)
    print('done ', WORKERS)
    WORKERS -= 1
    return results.append(time.time() - b)


async def main():
    global WORKERS
    args = get_args()
    data, sr = sf.read(args.wav_path, dtype='int16')
    assert sr == 16000
    duration = (len(data)) / 16000
    total_duration = duration * args.batch_size * args.requests
    print(f'{duration = }')
    data = data.tobytes()
    batch_data = [data] * args.batch_size

    tasks = []
    request_times = []
    failures = []
    begin = time.time()
    print(f'{args.requests = }, {args.concurrency = }')
    for _ in range(args.requests):
        task = asyncio.create_task(
                worker(batch_data, args.ws_uri, request_times, failures))
        tasks.append(task)
        if len(tasks) >= args.concurrency:
            t = tasks.pop(0)
            await t
    while WORKERS > 0:
        await asyncio.sleep(0.1)
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
