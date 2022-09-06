#!/usr/bin/env python3
# coding:utf-8

import json
import time
import hashlib
import asyncio
import argparse
import aiohttp
import soundfile as sf
import statistics

WORKERS = 0


def get_args():
    description = 'speed test of asr_api_server with callback'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-u', '--uri', required=True,
        default='http://127.0.0.1:8500/asr/v1/speech_rec',
        help="asr_api_server's uri")
    parser.add_argument(
        '-b', '--callback', required=True,
        default='http://127.0.0.1:8500/asr/v1/callback_test',
        help='callback url')
    parser.add_argument(
        '-c', '--concurrency', type=int, required=True,
        help='number of requests to make at a time')
    parser.add_argument(
        '-n', '--requests', type=int, required=True,
        help='number of requests to perform')
    args = parser.parse_args()
    return args


async def worker(audio_url, api, callback):
    task_id = hashlib.md5(audio_url.encode('utf8')).hexdigest()
    info = {
        'task_id': task_id,
        'enable_punctution_prediction': True,
        'file_path': audio_url,
        'callback_url': callback,
    }
    global WORKERS
    WORKERS += 1
    print('start ', WORKERS)
    async with aiohttp.ClientSession() as session:
        async with session.post(api, json=info) as resp:
            text = await resp.text()
            print(text)
    print('done ', WORKERS)
    WORKERS -= 1


def gen_audio_urls(count):
    # first run python -m http.server 8020 in the dir:
    # /aidata/audio/ahc_audio_annotated/merge_opus/
    pre = 'http://127.0.0.1:8020/'
    urls = []
    for i in range(1, count+1):
        url = f'{pre}{i}.opus'
        urls.append(url)
    return urls


async def main():
    global WORKERS
    args = get_args()
    audios = gen_audio_urls(args.requests)
    begin = time.time()
    tasks = []
    for audio_url in audios:
        task = asyncio.create_task(
                worker(audio_url, args.uri, args.callback))
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
    print('done', f'{request_time = }')


if __name__ == '__main__':
    asyncio.run(main())
