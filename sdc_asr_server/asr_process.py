#encoding: utf8
import asyncio
import json
import time
import numpy as np
import soundfile as sf
import aiohttp
from io import BytesIO
from sdc_asr_server import vad_gpvad as vad
from sdc_asr_server.logger import logger
from sdc_asr_server.ws_query import ws_rec


async def download(url):
    logger.info(f'download audio:{url}')
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=2) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    print('download data:', len(data))
                    msg = 'ok'
                else:
                    data = b''
                    msg = 'download audio url failed with status: {}'.format(resp.status)
    except Exception as e:
        data = b''
        msg = 'download audio url failed with exception: {}'.format(e)
    return data, msg


async def rec(audio_origin):
    b = time.time()
    # with open('z-origin.wav', 'wb') as f:
    #     f.write(audio_origin)
    segments, duration, samplerate = vad.vad(audio_origin)
    logger.info(f'vad time: {time.time()-b}, audio {duration=}')
    max_len = 0
    tasks = []
    i = 0
    for s in segments:
        i += 1
        # print('\t==== segm:', i, type(s), s.shape)
        d = len(s) / samplerate
        if d > max_len:
            max_len = d
        t = asyncio.create_task(ws_rec(s.astype('int16').tobytes()))
        tasks.append(t)
    texts = []
    for task in tasks:
        text = await task
        texts.append(text)
    timing = time.time() - b
    if not texts:
        texts = ['no speech detected']
    # else:
    #     text = ','.join(texts)
    # logger.info(f"rec text: {text}")
    logger.info(f'[{len(audio_origin)}], duration[{duration}] seconds, time use:{timing}, segment {max_len=}, segments:{i}')
    print(f'{len(texts) = }')
    return ','.join(texts)
