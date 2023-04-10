#!/usr/bin/env python3
# coding:utf-8

import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import np_to_triton_dtype
import soundfile as sf
import statistics
import numpy as np
import asyncio
import argparse
import time
import os

TRITON_FLAGS = {
    'url': "localhost:8001",
    'verbose': False,
    'model_name': 'infer_pipeline',
}


async def triton_rec(data: bytes, url: str, verbose: bool) -> dict:
    """

    :param data: int16字节音频数据
    :return:
    """
    start_time = time.time()
    text = ''
    samples = np.frombuffer(data, dtype='int16')
    samples = np.array([samples], dtype=np.float32)
    lengths = np.array([[len(samples)]], dtype=np.int32)
    protocol_client = grpcclient
    inputs = [
        protocol_client.InferInput(
            "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
        ),
        protocol_client.InferInput(
            "WAV_LENS", lengths.shape, np_to_triton_dtype(lengths.dtype)
        ),
    ]
    inputs[0].set_data_from_numpy(samples)
    inputs[1].set_data_from_numpy(lengths)
    outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]
    sequence_id = 10086
    # with grpcclient.InferenceServerClient(
    #         url=TRITON_FLAGS['url'], verbose=TRITON_FLAGS['verbose']
    # ) as triton_client:
    triton_client = grpcclient.InferenceServerClient(url=url, verbose=TRITON_FLAGS['verbose'])
    try:
        response = await triton_client.infer(
            TRITON_FLAGS['model_name'],
            inputs,
            request_id=str(sequence_id),
            outputs=outputs,
        )
        text = response.as_numpy("TRANSCRIPTS")[0].decode("utf-8")
    except Exception as e:
        print(e)

    time_cost = time.time() - start_time
    if verbose:
        print(time_cost, '||', text)
    return {
        'text': text,
        'time': time_cost
    }


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-u', '--url_triton',
        default='localhost:8001',
        help="triton server grpc url, default: 'localhost:8001'")
    parser.add_argument(
        '-w', '--wav_scp', required=True,
        help='path to wav_scp_file')
    parser.add_argument(
        '-t', '--trans', required=True,
        help='path to trans_text_file of wavs')
    parser.add_argument(
        '-s', '--save_to', required=True,
        help='path to save transcription')
    parser.add_argument(
        '-n', '--num_concurrence', type=int, required=True,
        help='num of concurrence for query')
    parser.add_argument(
        '-v', '--verbose', required=False, default=False,
        help='show logs')
    args = parser.parse_args()
    return args


async def run(wav_scp, args):
    tasks = []
    texts = []
    request_times = []
    for i, (_uttid, data) in enumerate(wav_scp):
        task = asyncio.create_task(triton_rec(data, args.url_triton, args.verbose))
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


def print_result(info):
    length = max([len(k) for k in info])
    for k, v in info.items():
        print(f'\t{k: >{length}} : {v}')


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
    cmd = f'tail -n 8 {cer_file}'
    print(cmd)
    os.system(cmd)
    print('done')


if __name__ == '__main__':
    args = get_args()
    asyncio.run(main(args))
