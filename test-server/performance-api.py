#!/usr/bin/env python3
# coding:utf-8

import os
import json
import time
import asyncio
import argparse
import aiohttp
import statistics


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-u', '--api_uri',
        default='http://127.0.0.1:8300/asr/v1/rec',
        help="asr_api_server's uri, default: http://127.0.0.1:8300/asr/v1/rec")
    parser.add_argument(
        '-w', '--wav_scp', required=True,
        help='path to wav_scp_file')
    parser.add_argument(
        '-t', '--trans', required=True,
        help='path to trans_text_file of wavs')
    parser.add_argument(
        '-i', '--ignore', required=True,
        help='path to ignore words file')
    parser.add_argument(
        '-s', '--save_to', required=True,
        help='path to save transcription')
    parser.add_argument(
        '-n', '--num_concurrence', type=int, required=True,
        help='num of concurrence for query')
    args = parser.parse_args()
    return args


def print_result(info):
    length = max([len(k) for k in info])
    for k, v in info.items():
        print(f'\t{k: >{length}} : {v}')


async def test_coro(api, audio_data, uttid, texts, times):
    headers = {
        'appkey': '123',
        'format': 'pcm',
    }
    begin = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(api, data=audio_data, headers=headers) as resp:
            text = await resp.text()
    time_cost = time.time() - begin
    text = json.loads(text)['text']
    texts.append(f'{uttid}\t{text}\n')
    times.append(time_cost)


async def main(args):
    wav_scp = []
    total_duration = 0
    with open(args.wav_scp) as f:
        for line in f:
            zz = line.strip().split()
            assert len(zz) == 2
            with open(zz[1], 'rb') as f:
                data = f.read()
            duration = (len(data) - 44)/2/16000
            total_duration += duration
            wav_scp.append((zz[0], data))
    print(f'{len(wav_scp) = }, {total_duration = }')

    tasks = []
    failed = 0
    texts = []
    request_times = []
    begin = time.time()
    for i, (_uttid, data) in enumerate(wav_scp):
        task = asyncio.create_task(
                test_coro(args.api_uri, data, _uttid, texts, request_times))
        tasks.append(task)
        if len(tasks) < args.num_concurrence:
            continue
        print((f'{i=}, start {args.num_concurrence} '
               f'queries @ {time.strftime("%m-%d %H:%M:%S")}'))
        task = tasks.pop(0)
        await task
    for uttid, task in tasks:
        await task
    request_time = time.time() - begin
    rtf = request_time / total_duration
    concur_info = {
        'failed': failed,
        'total_duration': total_duration,
        'request_time': request_time,
        'RTF': rtf,
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
    with open(args.save_to + '.log', 'w', encoding='utf8') as flog:
        flog.write(json.dumps(concur_info, indent=4))
        flog.write('\n')
        flog.write(json.dumps(request_info, indent=4))
    with open(args.save_to, 'w', encoding='utf8') as fsave:
        fsave.write(''.join(texts))
    # caculate CER
    cmd = (f'python ../test-model/compute-wer.py --char=1 --v=1 '
           f'--ig={args.ignore} '
           f'{args.trans} {args.save_to} > '
           f'{args.save_to}-test-{args.num_concurrence}.cer.txt')
    print(cmd)
    os.system(cmd)
    print('done')


if __name__ == '__main__':
    args = get_args()
    asyncio.run(main(args))
