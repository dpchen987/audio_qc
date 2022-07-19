# encoding: utf8
import re
import os
import asyncio
import time
import aiohttp
import requests
import traceback
from io import BytesIO
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
if os.environ.get('VAD_WEBRTC'):
    from asr_api_server.vad_webrtc import vad
    print('==='*20)
    print('Using VAD_WEBRTC ...')
    print('==='*20)
else:
    from asr_api_server.vad_gpvad import vad
    print('==='*20)
    print('Using VAD_GPVAD ...')
    print('==='*20)
from asr_api_server.logger import logger
from asr_api_server.ws_query import ws_rec


async def download(url):
    b = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=8) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    msg = 'ok'
                else:
                    data = b''
                    msg = f'download {url} failed with status: {resp.status}'
    except Exception as e:
        logger.error(e)
        data = b''
        msg = 'download audio url failed with exception: {}'.format(e)
    time_cost = time.time() - b
    logger.debug(f'{time_cost = } for downloading :{url}')
    return data, msg


async def rec_no_vad(audio_origin):
    b = time.time()
    bio = BytesIO(audio_origin)
    data, samplerate = sf.read(bio, dtype='int16')
    try:
        text = await ws_rec(data.astype('int16').tobytes())
        exception = 0
    except Exception as e:
        logger.debug(e)
        traceback.print_exc()
        text = ''
        exception = 1
    timing = time.time() - b
    logger.info(f'REC without VAD: time use:{timing}, {exception=}')
    text = re.sub(r'<.*?>', '', text)
    return text, exception


async def rec_vad(audio_origin):
    b = time.time()
    segments, duration, samplerate = vad(audio_origin)
    logger.info(f'vad time: {time.time()-b}, {duration=}, {len(segments) = }')
    max_len = 0
    tasks = []
    for s in segments:
        d = len(s) / samplerate
        if d > max_len:
            max_len = d
        t = asyncio.create_task(ws_rec(s))
        tasks.append(t)
    results = []
    exception = 0
    for task in tasks:
        result = {'err': ''}
        try:
            result['text'] = await task
        except Exception as e:
            traceback.print_exc()
            result['text'] = ''
            result['err'] = str(e)
            exception += 1
        results.append(result)
    timing = time.time() - b
    msg = (f'REC: duration: [{duration}] seconds, '
           f'time use:{timing}, max len of segment: {max_len}, '
           f'segments_count: {len(segments)}, {exception=}')
    logger.info(msg)
    text = ','.join([r['text'] for r in results])
    text = re.sub(r'<.*?>', '', text)
    logger.info(text)
    return text, exception


async def rec(audio_origin):
    return await rec_vad(audio_origin)
