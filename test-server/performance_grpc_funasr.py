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
import queue
import asyncio
import argparse
import grpc
import soundfile as sf
import statistics

import paraformer_pb2


class RecognizeStub:
    def __init__(self, channel):
        self.Recognize = channel.stream_stream(
                '/paraformer.ASR/Recognize',
                request_serializer=paraformer_pb2.Request.SerializeToString,
                response_deserializer=paraformer_pb2.Response.FromString,
                )


class FunAsrGrpcClient:
    def __init__(self, host, port, user='zz', language='zh-CN'):
        self.user = user
        self.lang = language
        self.channel = grpc.insecure_channel(f'{host}:{port}')

    def __del__(self,):
        # self.send(None, speaking=False, isEnd=True)
        self.channel.close()

    async def send(self, data, speaking=True, isEnd=False):
        self.stub = RecognizeStub(self.channel)
        req = paraformer_pb2.Request()
        if data:
            req.audio_data = data
        req.user = self.user
        req.language = self.lang
        req.speaking = speaking
        req.isEnd = isEnd
        q = queue.SimpleQueue()
        q.put(req)
        return self.stub.Recognize(iter(q.get, None))

    async def rec(self, data):
        response = await self.send(data, speaking=False)
        print(type(response))
        resp = response.next()
        text = ''
        b = time.time()
        if 'decoding' == resp.action:
            resp = response.next()
            if 'finish' == resp.action:
                text = json.loads(resp.sentence)['text']
        return {
                'text': text,
                'time': time.time() - b,
                }


async def send(channel, data, speaking, isEnd):
    stub = RecognizeStub(channel)
    req = paraformer_pb2.Request()
    if data:
        req.audio_data = data
    req.user = 'zz'
    req.language = 'zh-CN'
    req.speaking = speaking
    req.isEnd = isEnd
    q = queue.SimpleQueue()
    q.put(req)
    return stub.Recognize(iter(q.get, None))


async def grpc_rec(data, grpc_uri):
    with grpc.insecure_channel(grpc_uri) as channel:
        b = time.time()
        response = await send(channel, data, False, False)
        resp = response.next()
        text = ''
        if 'decoding' == resp.action:
            resp = response.next()
            if 'finish' == resp.action:
                text = json.loads(resp.sentence)['text']
        response = await send(channel, None, False, True)
        return {
                'text': text,
                'time': time.time() - b,
                }


def get_args():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(
    #     '--host', required=True,
    #     help="grpc server")
    # parser.add_argument(
    #     '--port', required=True,
    #     type=int,
    #     help="grpc server")
    parser.add_argument(
        '--wav_scp', required=True,
        help='path to wav_scp_file')
    parser.add_argument(
        '--trans', required=True,
        help='path to trans_text_file of wavs')
    parser.add_argument(
        '--save_to', required=True,
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

uris = [
        '127.0.0.1:9900',
        '127.0.0.1:9901',
        ]
uri_idx = 0


async def run(wav_scp, args):
    global uris, uri_idx
    tasks = []
    texts = []
    request_times = []
    for i, (_uttid, data) in enumerate(wav_scp):
        # uri = f'{args.host}:{args.port}'
        uri_idx = uri_idx % len(uris)
        uri = uris[uri_idx]
        uri_idx += 1
        task = asyncio.create_task(grpc_rec(data, uri))
        tasks.append((_uttid, task))
        if len(tasks) < args.num_concurrence:
            continue
        if i % args.num_concurrence == 0:
            print((f'{i=}, @ {time.strftime("%m-%d %H:%M:%S")}'))
        for uttid, task in tasks:
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


async def main(args):
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


async def test():
    # fc = FunAsrGrpcClient('127.0.0.1', 9900)
    # t = await fc.rec(wav.tobytes())
    # print(t)
    wav, _ = sf.read('z-10s.wav', dtype='int16')
    uri = '127.0.0.1:9900'
    res = await grpc_rec(wav.tobytes(), uri)
    print(res)


if __name__ == '__main__':
    # asyncio.run(test())
    args_ = get_args()
    asyncio.run(main(args_))
