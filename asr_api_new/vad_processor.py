import time
import base64
from .vad_gpvad import vad
from .logger import logger
from .downloader import download
from .mqconn import mq
from .config import CONF


async def to_cache(segments, samplerate, audio_info):
    real = 0 
    if segments:
        real = sum([len(s) / samplerate for s in segments])
    data = [s.tobytes() for s in segments]
    for d in data:
        if CONF['joiner'] not in d:
            continue
        logger.error(f'=== find {CONF["joiner"]} in {audio_info.task_id = }')
    item = {
        'data': CONF['joiner'].join(data),
        'url': audio_info.callback_url,
        'task_id': audio_info.task_id,
    }
    await mq.put(item)
    return real


async def speech_vad(audio_info):
    '''识别语音为文本，接收语音数据audio-url参数，返回转译文本
    '''
    vad_result = {"code": 0, "data": "0", "msg": "success"}
    # 音频下载
    if audio_info.trans_type == 1:
        audio, msg = await download(audio_info.file_path)
    else:
        audio = base64.b64decode(audio_info.file_content)
        msg = 'ok' if audio else 'audio file is empty !!!'
    if msg != 'ok':
        vad_result['code'] = 4003
        vad_result['msg'] = msg
        return vad_result
    if not audio:
        vad_result['code'] = 4002
        vad_result['msg'] = 'no audio data'
        return vad_result
    # 音频vad
    b = time.time()
    try:
        segments, duration, samplerate = vad(audio)
    except Exception as e:
        vad_result['code'] = 4004
        vad_result['msg'] = str(e)
        return vad_result
    logger.info(f'vad timing: {time.time() - b}')
    b = time.time()
    try:
        real = await to_cache(segments, samplerate, audio_info)
    except Exception as e:
        vad_result['code'] = 4005
        vad_result['msg'] = str(e)
        return vad_result
    logger.info(f'to_cache timing: {time.time() - b}')
    vad_result['data'] = str(round(real, 3))
    return vad_result
