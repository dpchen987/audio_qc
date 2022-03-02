#encoding: utf8
import asyncio
import json
import time
import numpy as np
import websockets
import soundfile as sf
import librosa
import aiohttp
from io import BytesIO
from sdc_asr_server import vad_gpvad as vad
from sdc_asr_server.logger import logger
from sdc_asr_server.config import ARGS


WS_INDEX = 0
WS_START = json.dumps({
    'signal': 'start',
    'nbest': 1,
    'continuous_decoding': False,
})
WS_END = json.dumps({
    'signal': 'end'
})


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


def get_ws():
    if len(ARGS.ws) == 1:
        return ARGS.ws[0]
    global WS_INDEX
    idx = WS_INDEX / len(ARGS.ws)
    WS_INDEX = idx + 1
    return ARGS.ws[idx]


async def ws_rec(data):
    ws = get_ws()
    # logger.info(f'connect to {ws}')
    texts = []
    conn = await websockets.connect(ws)
    # async with websockets.connect(ws) as conn:
    # step 1: send start
    await conn.send(WS_START)
    ret = await conn.recv()
    # step 2: send audio data
    await conn.send(data)
    # step 3: send end
    await conn.send(WS_END)
    # step 3: receive result
    i = 0
    while 1:
        i += 1
        ret = await conn.recv()
        # print('ws recv loop', i, ret)
        ret = json.loads(ret)
        if ret['type'] == 'final_result':
            nbest = json.loads(ret['nbest'])
            text = nbest[0]['sentence']
            texts.append(text)
        elif ret['type'] == 'speech_end':
            # print('=======', ret)
            break
    try:
        await conn.close()
    except Exception as e:
        logger.error(e)
    return ''.join(texts)


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
