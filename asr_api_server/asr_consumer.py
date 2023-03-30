#!/usr/bin/env python
# coding:utf-8


import os
import json
import asyncio
import aiohttp
import base64
from asr_api_server import asr_process
from asr_api_server.logger import logger
from asr_api_server import config
from asr_api_server.data_model.api_model import AudioInfo

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
        # 不管回调是否成功，都删除processing_set中的task
        config.processing_set.discard(audio_info.task_id)
        # if audio_info.file_content == 'processed' and os.path.exists(audio_info.file_path):
        #     os.remove(audio_info.file_path)
        try:
            # 回调接口调用
            logger.info(f"{Callback_param}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url=audio_info.callback_url, data=json.dumps(Callback_param, ensure_ascii=False), headers={'content-type': 'application/json'}) as resp:
                    html = await resp.text()
            resp_dt = json.loads(html)
            if resp_dt["code"] == 0:
                # 识别完成，清理数据库
                # if os.path.exists(audio_info.file_path): os.remove(audio_info.file_path)
                config.url_db.Delete(audio_info.task_id.encode())
            else:
                logger.info("回调失败！！")
                logger.info(f"{resp_dt}")
        except Exception as e:
            logger.info("回调失败！！")
            logger.exception(e)
