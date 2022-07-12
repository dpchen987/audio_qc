# encoding: utf8
import re
import asyncio
import time
import aiohttp
import traceback
from io import BytesIO
import soundfile as sf
from asr_api_server import vad_gpvad as vad
from asr_api_server.logger import logger
from asr_api_server.ws_query import ws_rec


async def download(url):
    logger.info(f'download audio:{url}')
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=2) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    msg = 'ok'
                else:
                    data = b''
                    msg = 'download audio url failed with status: {}'.format(resp.status)
    except Exception as e:
        data = b''
        msg = 'download audio url failed with exception: {}'.format(e)
    return data, msg


async def rec_no_vad(audio_origin):
    b = time.time()
    # with open('z-origin.wav', 'wb') as f:
    #     f.write(audio_origin)
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
    # with open('z-origin.wav', 'wb') as f:
    #     f.write(audio_origin)
    segments, duration, samplerate = vad.vad(audio_origin)
    logger.info(f'vad time: {time.time()-b}, audio {duration=}')
    max_len = 0
    tasks = []
    segments_count = 0
    for s in segments:
        segments_count += 1
        d = len(s) / samplerate
        if d > max_len:
            max_len = d
        t = asyncio.create_task(ws_rec(s.astype('int16').tobytes()))
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
           f'segments_count: {segments_count}, {exception=}')
    logger.info(msg)
    text = ','.join([r['text'] for r in results])
    text = re.sub(r'<.*?>', '', text)
    return text, exception


async def rec(audio_origin):
    return await rec_no_vad(audio_origin)
