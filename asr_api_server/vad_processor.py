import time
import asyncio
import base64
import json
from .vad_gpvad import vad_duration
from .logger import logger
from asr_api_server import asr_process

VAD_NUM = asyncio.Semaphore(100)


async def speech_vad(audio_info):
    '''识别语音为文本，接收语音数据audio-url参数，返回转译文本
    '''
    vad_result = {"code":0, "data": "0", "msg":"success"}
    try:
        async with VAD_NUM:
            # 音频下载
            if audio_info.trans_type == 1:
                audio, msg = await asr_process.download(audio_info.file_path)
            else:
                # with open(audio_info.file_path, 'rb') as f:
                #     audio = f.read()
                b = time.time()
                audio = base64.b64decode(audio_info.file_content)
                msg = 'ok' if audio else 'audio file is empty !!!'
            if msg != 'ok':
                vad_result['code'] = 4003
                vad_result['msg'] = msg
                return
            if not audio:
                vad_result['code'] = 4002
                vad_result['msg'] = 'no audio data'
                return
            # 音频vad
            b = time.time()
            # pre vad
            total = vad_duration(audio, prevad=True)
            if total < 1.5:
                # full vad
                total = max(total, vad_duration(audio, prevad=False))
            logger.info(f'vad time: {time.time() - b}, {total = }')
            vad_result['data'] = json.dumps(total)
    except Exception as e:
        logger.exception(e)
        vad_result['code'] = 4000
        vad_result['msg'] = str(e)
    return vad_result
