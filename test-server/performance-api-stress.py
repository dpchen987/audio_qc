#!/usr/bin/env python3
# coding:utf-8

import os
import base64
import json
import time
import asyncio
import argparse
import aiohttp
import statistics
import soundfile as sf

api_indx = 0


def get_api():
    global api_indx
    apis = [
        'http://127.0.0.1:8300/asr/v1/speech_rec',
        'http://127.0.0.1:8400/asr/v1/speech_rec',
    ]
    api = apis[api_indx]
    api_indx += 1 
    api_indx %= len(apis)
    return api


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-u', '--api_uri',
        default='http://127.0.0.1:8300/asr/v1/speech_rec',
        help="asr_api_server's uri, default: http://127.0.0.1:8300/asr/v1/speech_rec")
    parser.add_argument(
        '-d', '--dir_audio', required=True,
        help='dir to file of audio')
    parser.add_argument(
        '-c', '--concurrence', type=int, required=True,
        help='num of concurrence for query')
    parser.add_argument(
        '-n', '--num_query', type=int, default=0,
        help='num of total query')
    args = parser.parse_args()
    return args


async def test_coro(api, taskid, data):
    b64 = base64.b64encode(data).decode('utf8')
    query = {
        'task_id': taskid,
        'callback_url': 'http://192.168.10.10:8304/asr/v1/callBack_test',
        'enable_punctution_prediction': True,
        'file_path': 'nopath',
        'file_content': b64,
        'file_type': 'opus',
        'trans_type': 2,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(api, json=query) as resp:
            text = await resp.text()

def get_files(root_dir):
    paths = []
    for root, dirs, files in os.walk(root_dir):
        for fi in files:
            if not fi.endswith('.opus'):
                continue
            path = os.path.join(root, fi)
            paths.append(path)
    print(f'{len(paths) = }')
    return paths


async def main(args):
    audios = get_files(args.dir_audio)
    audios.sort()
    print(f'{len(audios) = }')
    if args.num_query:
        audios = audios[:args.num_query]
    print(f'{len(audios) = }')

    tasks = set()
    result = []
    begin = time.time()
    for i, path in enumerate(audios):
        with open(path, 'rb') as f:
            data = f.read()
        now = time.strftime('%Y-%m-%d_%H:%M:%S')
        task_id = now + f'--{i:06}'
        result.append(task_id)
        api_uri = get_api()
        task = asyncio.create_task(
                test_coro(api_uri, task_id, data))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        if len(tasks) < args.concurrence:
            continue
        await asyncio.sleep(0.1)
        if i % args.concurrence == 0:
            print((f'{i=}, start {args.concurrence} '
                   f'queries @ {time.strftime("%m-%d %H:%M:%S")}'))
    while tasks:
        await asyncio.sleep(0.1)
    with open(f'{now}.log', 'w') as f:
        json.dump(result, f, indent=2)
    print('done', time.time() - begin)


if __name__ == '__main__':
    args = get_args()
    asyncio.run(main(args))
