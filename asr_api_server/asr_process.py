# encoding: utf8
import re
import os
import asyncio
import time
import aiohttp
import traceback
from io import BytesIO
import soundfile as sf
from .logger import logger
from .easytimer import Timer

eztimer = Timer(logger)
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
model_dir = os.environ.get('ASR_LOCAL_MODEL')
if model_dir:
    asr_type = 'local'
    from .recognize_onnx import AsrOnnx
    print('==='*20)
    print('Using LOCAL MODEL:', model_dir)
    print('==='*20)
    asronnx = AsrOnnx(model_dir)
else:
    asr_type = 'ws'
    print('=== No environ: ASR_LOCAL_MODEL, Using [ws] decoder ===')
    if os.environ.get('ASR_BATCH'):
        asr_type = 'ws_batch'
        print('\t=== Using [ws_batch] decoder ===')
        from asr_api_server.ws_query_batch import ws_rec
    else:
        from asr_api_server.ws_query import ws_rec


async def download(url):
    b = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=80) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    msg = 'ok'
                else:
                    data = b''
                    msg = f'download {url} failed with status: {resp.status}'
    except Exception as e:
        logger.exception(e)
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


def rec_vad_local(audio_origin):
    eztimer.begin('vad')
    segments, duration, samplerate = vad(audio_origin)
    eztimer.end('vad')
    texts = asronnx.rec(segments)
    return '，'.join(texts), 0


async def rec_vad_ws(audio_origin):
    b = time.time()
    segments, duration, samplerate = vad(audio_origin)
    logger.info(f'vad time: {time.time()-b}, {duration=}, {len(segments) = }')
    max_len = 0
    tasks = []
    for s in segments:
        d = len(s) / samplerate
        if d > max_len:
            max_len = d
        t = asyncio.create_task(ws_rec(s.tobytes()))
        tasks.append(t)
    results = []
    exception = 0
    exception_ls = []
    for task in tasks:
        result = {'err': ''}
        try:
            result['text'] = await task
        except Exception as e:
            traceback.print_exc()
            result['text'] = ''
            result['err'] = str(e)
            exception += 1
            if repr(e) not in exception_ls: exception_ls.append(repr(e))
        results.append(result)
    timing = time.time() - b
    msg = (f'REC: duration: [{duration}] seconds, '
           f'time use:{timing}, max len of segment: {max_len}, '
           f'segments_count: {len(segments)}, {exception=}')
    logger.info(msg)
    text = ','.join([r['text'] for r in results])
    text = re.sub(r'<.*?>', '', text)
    logger.info(text)
    return text, f"{exception} exceptions : {', '.join(exception_ls)}"


async def rec_vad_ws_batch(audio_origin):
    eztimer.begin('vad')
    segments, duration, samplerate = vad(audio_origin)
    eztimer.end('vad')
    data = [s.tobytes() for s in segments]
    eztimer.begin('[batch recognize]')
    try:
        texts = await ws_rec(data)
        if not texts:
            exception = 1
        else:
            exception = 0
    except:
        traceback.print_exc()
        texts = []
        exception = 1
    eztimer.end('[batch recognize]')
    text = re.sub(r'<.*?>', '', '，'.join(texts))
    return text, exception


async def rec(audio_origin):
    if asr_type == 'local':
        return rec_vad_local(audio_origin)
    if asr_type == 'ws_batch':
        return await rec_vad_ws_batch(audio_origin)
    return await rec_vad_ws(audio_origin)
