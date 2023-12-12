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


WS_START = json.dumps({
    'chunk_size': [5, 0, 5],
    'wav_name': 'yshy',
    'is_speaking': True,
})

WS_END = json.dumps({
    'is_speaking': False
})


async def ws_rec(data, ws_uri):
    begin = time.time()
    conn = await websockets.connect(ws_uri, ping_timeout=200)
    # step 1: send start
    await conn.send(WS_START)
    # step 2: send audio data
    await conn.send(data)
    # step 3: send end
    await conn.send(WS_END)
    # step 4: receive result
    texts = []
    ret = await conn.recv()
    print(ret)
    ret = json.loads(ret)
    text = ret['text']
    # step 5: close
    try:
        await conn.close()
    except Exception as e:
        # this except has no effect, just log as debug
        # it seems the server does not send close info, maybe
        print(e)
    time_cost = time.time() - begin
    return {
        'text': text,
        'time': time_cost,
    }


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-f', '--file', default='',
        help="run for single file"
    )
    parser.add_argument(
        '-u', '--ws_uri', default='ws://127.0.0.1:8889',
        help="websocket_server_main's uri, e.g. ws://127.0.0.1:8889")
    parser.add_argument(
        '-w', '--wav_scp',
        help='path to wav_scp_file')
    parser.add_argument(
        '-t', '--trans',
        help='path to trans_text_file of wavs')
    parser.add_argument(
        '-s', '--save_to',
        help='path to save transcription')
    parser.add_argument(
        '-n', '--num_concurrence', type=int, default=1,
        help='num of concurrence for query')
    parser.add_argument(
        '-b', '--batch', type=int, default=1,
        help='batch size for run_batch')
    args = parser.parse_args()
    return args


def print_result(info):
    length = max([len(k) for k in info])
    for k, v in info.items():
        print(f'\t{k: >{length}} : {v}')


uris = [
    'ws://127.0.0.1:8888',
    'ws://127.0.0.1:8889',
]


async def run(wav_scp, args):
    tasks = []
    texts = []
    request_times = []
    for i, (_uttid, data) in enumerate(wav_scp):
        # idx = i % 2
        # uri = uris[idx]
        uri = args.ws_uri
        task = asyncio.create_task(ws_rec(data, uri))
        tasks.append((_uttid, task))
        if len(tasks) < args.num_concurrence:
            continue
        if i % args.num_concurrence == 0:
            print((f'{i=}, @ {time.strftime("%m-%d %H:%M:%S")}'))
        uttid, task = tasks.pop(0)
        result = await task
        texts.append(f'{uttid}\t{result["text"]}\n')
        request_times.append(result['time'])
    if tasks:
        for uttid, task in tasks:
            result = await task
            texts.append(f'{uttid}\t{result["text"]}\n')
            request_times.append(result['time'])
    with open(args.save_to, 'w', encoding='utf8') as fsave:
        fsave.write(''.join(texts))
    return request_times


async def run_batch(wav_scp, args):
    def batchit(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i: i + batch_size]

    tasks = []
    texts = []
    request_times = []
    for i, batch in enumerate(batchit(wav_scp, args.batch)):
        uttids = [b[0] for b in batch]
        data = [b[1] for b in batch]
        task = asyncio.create_task(ws_rec_batch(data, args.ws_uri))
        tasks.append((uttids, task))
        if len(tasks) < args.num_concurrence:
            continue
        if i % args.num_concurrence == 0:
            print((f'{i=}, @ {time.strftime("%m-%d %H:%M:%S")}'))
        uttids, task = tasks.pop(0)
        result = await task
        for j in range(len(uttids)):
            texts.append(f'{uttids[j]}\t{result["texts"][j]}\n')
        request_times.append(result['time'])
    if tasks:
        for uttids, task in tasks:
            result = await task
            for j in range(len(uttids)):
                texts.append(f'{uttids[j]}\t{result["texts"][j]}\n')
            request_times.append(result['time'])
    with open(args.save_to, 'w', encoding='utf8') as fsave:
        fsave.write(''.join(texts))
    return request_times


async def main(args):
    if args.file:
        wav, sr = sf.read(args.file, dtype='int16')
        b = time.time()
        ret = await ws_rec(wav.tobytes(), args.ws_uri)
        e = time.time()
        print(ret)
        print(e-b)
        return

    # 1. read data
    wav_scp = []
    total_duration = 0
    with open(args.wav_scp) as f:
        for line in f:
            zz = line.strip().split()
            assert len(zz) == 2
            data, sr = sf.read(zz[1], dtype='int16')
            assert sr == 16000
            duration = (len(data)) / 16000
            total_duration += duration
            wav_scp.append((zz[0], data.tobytes()))
    print(f'{len(wav_scp) = }, {total_duration = }')

    # 2. run
    begin = time.time()
    if args.batch > 1:
        print('runing batch...')
        request_times = await run_batch(wav_scp, args)
    else:
        request_times = await run(wav_scp, args)

    # 3. result
    print('printing test result')
    run_time = time.time() - begin
    rtf = run_time / total_duration
    print('For all concurrence:')
    print_result({
        'total_duration': total_duration,
        'run_time': run_time,
        'RTF': rtf,
    })
    print('For one request:')
    print_result({
        'mean': statistics.mean(request_times),
        'median': statistics.median(request_times),
        'max_time': max(request_times),
        'min_time': min(request_times),
    })
    # caculate CER
    cer_file = f'{args.save_to}-test-{args.num_concurrence}.cer.txt'
    cmd = (f'python compute-wer.py --char=1 --v=1 --ig=cer-ignore.txt '
           f'{args.trans} {args.save_to} > {cer_file}')
    print(cmd)
    os.system(cmd)
    cmd = f'tail {cer_file}'
    print(cmd)
    os.system(cmd)
    print('done')


if __name__ == '__main__':
    args_ = get_args()
    asyncio.run(main(args_))
