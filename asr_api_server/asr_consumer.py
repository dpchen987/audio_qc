#!/usr/bin/env python
# coding:utf-8


import os
import json
import asyncio
import aiohttp
from asr_api_server import asr_process
from asr_api_server.logger import logger
from asr_api_server import config
from asr_api_server.data_model.api_model import AudioInfo, CallBackParam

COUNTER_MAX = int(round(os.cpu_count() / 2, 0))
COUNTER = 0
ASR_NUM = None
CALLBACK_URL = ""


async def asy_timer():
    '''定期检查是否有未执行的任务，并处理
    初始API的CALLBACK_URL位空，有调用后CALLBACK_URL取回调url
    只补调一次
    '''
    await asyncio.sleep(60)
    global CALLBACK_URL
    logger.info("-----Regular inspection tasks execution ...-----")
    logger.info(f"CallBack url:{CALLBACK_URL}")
    logger.info(f"tasks under processing: {config.processing_set}")
    for task_id, audio_url in config.url_db.RangeIter():
        if task_id.decode() not in config.processing_set and CALLBACK_URL:
            audio_info = AudioInfo(
                    task_id=task_id.decode(),
                    file_path=audio_url.decode(), callback_url=CALLBACK_URL)
            asyncio.create_task(speech_recognize(audio_info))
            logger.info(f"Recreate task：-----{audio_info.task_id} ！！！----")
            config.processing_set.add(audio_info.task_id)
            config.url_db.Delete(audio_info.task_id.encode())
    asyncio.create_task(asy_timer())


async def speech_recognize(audio_info):
    '''识别语音为文本，接收语音数据audio-url参数，返回转译文本
    '''
    global COUNTER
    global CALLBACK_URL
    global ASR_NUM
    if ASR_NUM is None:
        ASR_NUM = asyncio.Semaphore(COUNTER_MAX)
    # while COUNTER > COUNTER_MAX:
    #     print(f'waiting in queque, cocurrency: {COUNTER}')
    #     await asyncio.sleep(1)
    Callback_param = {"task_id": audio_info.task_id, "code":0, "content": '', "err_msg":"success"}
#     if not CALLBACK_URL: CALLBACK_URL = audio_info.callback_url
    CALLBACK_URL = audio_info.callback_url
    try:
        COUNTER += 1
        async with ASR_NUM:
            print('=============== asr_consumer counter ', COUNTER)
            # 音频下载
            audio, msg = await asr_process.download(audio_info.file_path)
            if msg != 'ok':
                Callback_param['code'] = 4003
                Callback_param['err_msg'] = msg
                COUNTER -= 1
                return
            if not audio:
                Callback_param['code'] = 4002
                Callback_param['err_msg'] = 'no audio data'
                COUNTER -= 1
                return
            # 音频转译
            text, exception = await asr_process.rec(audio)
            if exception:
                Callback_param['code'] = 4004
                Callback_param['err_msg'] = exception
            Callback_param['content'] = text
            COUNTER -= 1
    except Exception as e:
        logger.exception(e)
        Callback_param['code'] = 4000
        Callback_param['err_msg'] = str(e)
        COUNTER -= 1
    finally:
        # 不管回调是否成功，都删除processing_set中的task
        config.processing_set.discard(audio_info.task_id)
        try:
            # 回调接口调用
            logger.info(f"{Callback_param}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url=audio_info.callback_url, data=json.dumps(Callback_param, ensure_ascii=False), headers={'content-type': 'application/json'}) as resp:
                    html = await resp.text()
            resp_dt = json.loads(html)
            if resp_dt["code"] == 0:
                # 识别完成，清理数据库
                config.url_db.Delete(audio_info.task_id.encode())
            else:
                logger.info("回调失败！！")
                logger.info(f"{resp_dt}")
        except Exception as e:
            logger.info("回调失败！！")
            logger.exception(e)
