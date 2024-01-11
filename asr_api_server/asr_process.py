# encoding: utf8
import re
import os
import asyncio
import time
import traceback
from io import BytesIO
import soundfile as sf
from .logger import logger
from .easytimer import Timer
from .config import CONF
from .downloader import download

eztimer = Timer(logger)
if os.environ.get('ASR_VAD_WEBRTC'):
    from asr_api_server.vad_webrtc import vad

    print('===' * 20)
    print('Using ASR_VAD_WEBRTC ...')
    print('===' * 20)
else:
    from asr_api_server.vad_gpvad import vad

    print('===' * 20)
    print('Using ASR_VAD_GPVAD ...')
    print('===' * 20)
model_dir = os.environ.get('ASR_LOCAL_MODEL')
if model_dir:
    asr_type = 'local'
    from .recognize_onnx import AsrOnnx

    print('===' * 20)
    print('Using LOCAL MODEL:', model_dir)
    print('===' * 20)
    asronnx = AsrOnnx(model_dir)
else:
    asr_type = 'ws'
    print('=== No environ: ASR_LOCAL_MODEL, Using [ws] decoder ===')
    if os.environ.get('ASR_RUN_BATCH'):
        asr_type = 'ws_batch'
        print('\t=== Using [ws_run_batch] decoder ===')
        from asr_api_server.ws_query_batch import ws_rec
    else:
        from asr_api_server.ws_query import ws_rec


async def rec_no_vad(audio_origin, dtype='int16'):
    b = time.time()
    bio = BytesIO(audio_origin)
    data, samplerate = sf.read(bio, dtype=dtype)
    try:
        text = await ws_rec(data.astype(dtype).tobytes())
        exception = 0
    except Exception as e:
        logger.debug(e)
        traceback.print_exc()
        text = ''
        exception = 1
    timing = time.time() - b
    text = re.sub(r'<.*?>', '', text)
    logger.info(f'REC without VAD. || REC Cost Time(s): {timing} || Exception: {exception} || Text: {text}')
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
    logger.info(f'vad time: {time.time() - b}, {duration=}, {len(segments) = }')
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
        result['text'], message = await task
        result['err'] = message
        if message:
            exception += 1
            exception_ls.append(message)
        results.append(result)
    timing = time.time() - b
    exception_info = f"{exception} exceptions : {', '.join(exception_ls)}" if exception else ''
    msg = (f'REC: duration: [{duration}] seconds, '
           f'time use:{timing}, max len of segment: {max_len}, '
           f'segments_count: {len(segments)}, error: {exception_info}')
    logger.info(msg)
    text = ','.join([r['text'] for r in results])
    text = re.sub(r'<.*?>', '', text)
    return text, exception


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
    if CONF['use_vad']:
        if asr_type == 'local':
            return rec_vad_local(audio_origin)
        if asr_type == 'ws_batch':
            return await rec_vad_ws_batch(audio_origin)
        return await rec_vad_ws(audio_origin)
    else:
        return await rec_no_vad(audio_origin)


if __name__ == '__main__':
    import sys
    fp = sys.argv[1]
    fname = fp.split('/')[-1]
    data = open(fp, 'rb').read()
    rest = asyncio.run(rec(data))
    print(rest)
