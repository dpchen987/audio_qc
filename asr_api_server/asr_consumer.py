#!/usr/bin/env python
# coding:utf-8


import os
import time
import json
import asyncio
import aiohttp
import base64
from asr_api_server import asr_process
from asr_api_server.logger import logger
from asr_api_server import config
from asr_api_server.data_model.api_model import AudioInfo
from asr_api_server.vad_gpvad import vad_duration

ASR_NUM = None
CALLBACK_URL = ""


async def asy_timer():
    '''定期检查是否有未执行的任务，并处理
    初始API的CALLBACK_URL位空，有调用后CALLBACK_URL取回调url
    只补调一次
    '''
    await asyncio.sleep(600)
    global CALLBACK_URL
    logger.info("-----Regular inspection tasks execution ...-----")
    logger.info(f"CallBack url:{CALLBACK_URL}")
    # logger.info(f"tasks under processing: {config.processing_set}")
    for task_id, inf_dict in config.url_db.RangeIter():
        if task_id.decode() not in config.processing_set and CALLBACK_URL:
            try:
                audio_info = AudioInfo(**json.loads(inf_dict.decode()))
                # audio_info.file_content = 'processed'
                asyncio.create_task(speech_recognize(audio_info))
                logger.info(f"Recreate task：-----{task_id.decode()} ！！！----")
                config.processing_set.add(task_id.decode())
                config.url_db.Delete(task_id)
            except Exception as e:
                logger.exception(f"asy_timer: {e}")
    asyncio.create_task(asy_timer())


async def speech_recognize(audio_info):
    '''识别语音为文本，接收语音数据audio-url参数，返回转译文本
    '''
    global CALLBACK_URL
    global ASR_NUM
    logger.info(f"task: {audio_info.task_id} enter Coroutine !-!")
    if ASR_NUM is None:
        ASR_NUM = asyncio.Semaphore(config.CONF['concurrency'])
    Callback_param = {"task_id": audio_info.task_id, "code":0, "content": '', "err_msg":"success"}
    CALLBACK_URL = audio_info.callback_url
    try:
        async with ASR_NUM:
            # 音频下载
            if audio_info.trans_type == 1:
                audio, msg = await asr_process.download(audio_info.file_path)
            else:
                # with open(audio_info.file_path, 'rb') as f:
                #     audio = f.read()
                audio = base64.b64decode(audio_info.file_content)
                msg = 'ok' if audio else 'audio file is empty !!!'
            if msg != 'ok':
                Callback_param['code'] = 4003
                Callback_param['err_msg'] = msg
                return
            if not audio:
                Callback_param['code'] = 4002
                Callback_param['err_msg'] = 'no audio data'
                return
            # 音频转译
            text, exception = await asr_process.rec(audio)
            if exception:
                Callback_param['code'] = 4004
                Callback_param['err_msg'] = exception
            Callback_param['content'] = text
    except Exception as e:
        logger.exception(e)
        Callback_param['code'] = 4000
        Callback_param['err_msg'] = str(e)
    finally:
        html = 'default'
        begin = time.time()
        try:
            # 回调接口调用
            logger.info(f"{Callback_param}")
            timeout = aiohttp.ClientTimeout(total=300)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url=audio_info.callback_url, data=json.dumps(Callback_param, ensure_ascii=False), headers={'content-type': 'application/json'}) as resp:
                    html = await resp.text()
            end = time.time()
            logger.info(f"CallBack time cost: {end-begin}")
            resp_dt = json.loads(html)
            if resp_dt["code"] == 0:
                # 识别完成，清理数据库
                # if os.path.exists(audio_info.file_path): os.remove(audio_info.file_path)
                config.url_db.Delete(audio_info.task_id.encode())
            else:
                logger.info(f"回调失败！！{resp_dt}")
                logger.info(f"{html}")
        except Exception as e:
            end = time.time()
            logger.info(f"回调失败！！time cost: {end-begin}, get response: {html}")
            logger.exception(e)
        finally:
            # 不管回调是否成功，都删除processing_set中的task
            config.processing_set.discard(audio_info.task_id)

VAD_NUM = asyncio.Semaphore(10)
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
            total = vad_duration(audio)
            logger.info(f'vad time: {time.time() - b}, {total = }')
            vad_result['data'] = json.dumps(total)
    except Exception as e:
        logger.exception(e)
        vad_result['code'] = 4000
        vad_result['msg'] = str(e)
    return vad_result
