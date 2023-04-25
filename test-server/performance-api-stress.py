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


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-u', '--api_uri',
        default='http://127.0.0.1:8300/asr/v1/speech_rec',
        help="asr_api_server's uri, default: http://127.0.0.1:8300/asr/v1/speech_rec")
    parser.add_argument(
        '-f', '--audio_list', required=True,
        help='path to file of audio url list')
    parser.add_argument(
        '-c', '--concurrence', type=int, required=True,
        help='num of concurrence for query')
    parser.add_argument(
        '-n', '--num_query', type=int, default=0,
        help='num of total query')
    args = parser.parse_args()
    return args


async def test_coro(api, taskid, audio_url, result):
    begin = time.time()
    query = {
            'task_id': taskid,
            'callback_url': 'http://127.0.0.1:8304/asr/v1/callBack_test',
            'enable_punctution_prediction': True,
            'file_path': audio_url,
            'file_content': 'abc',
            'file_type': 'opus',
            'trans_type': 1,
            }
    async with aiohttp.ClientSession() as session:
        async with session.post(api, json=query) as resp:
            text = await resp.text()
    end = time.time()
    result.append({
        'taskid': taskid,
        'begin': begin,
        'end': end,
        'text': text
        })


async def main(args):
    audio_urls = []
    with open(args.audio_list) as f:
        for line in f:
            zz = line.strip()
            if not zz.startswith('http'):
                print('bad', line)
                continue
            audio_urls.append(zz)
    print(f'{len(audio_urls) = }')
    tasks = set()
    result = []
    begin = time.time()
    now = time.strftime('%Y-%m-%d_%H:%M:%S')
    for i, audio_url in enumerate(audio_urls):
        if args.num_query and args.num_query < i:
            break
        task_id = now + f'{i:012}'
        task = asyncio.create_task(
                test_coro(args.api_uri, task_id, audio_url, result))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        if len(tasks) < args.concurrence:
            continue
        if i % args.concurrence == 0:
            print((f'{i=}, start {args.concurrence} '
                   f'queries @ {time.strftime("%m-%d %H:%M:%S")}'))
        await asyncio.sleep(0.05)
        while len(tasks) >= args.concurrence:
            await asyncio.sleep(0.05)
    while tasks:
        await asyncio.sleep(0.1)
    with open(f'{now}.log', 'w') as f:
        json.dump(result, f, indent=2)
    print('done', time.time() - begin)


if __name__ == '__main__':
    args = get_args()
    asyncio.run(main(args))
