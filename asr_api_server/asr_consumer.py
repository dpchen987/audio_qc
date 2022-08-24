#!/usr/bin/env python
# coding:utf-8


import os
import asyncio
from asr_api_server import asr_process
from asr_api_server.logger import logger
from asr_api_server import config

async def worker(url_audio, url_callback):
    '''
    1. call websocket_server
    2. send result to url_callback
    3. delete url_audio from db
    '''
    for item in db.RangeIter():
        if COUNTER < max_num:
            creat_task()
            COUNTER += 1
            break




async def consume():
    concurrency = os.cpu_count() // 2
    print(f'set {concurrency = }')
    while 1:
        print('consuming ...')
        # for _ in range(concurrency):
        #     url_audio, url_callback = get_from_db()
        #     asyncio.creat_task(worker(url_audio, url_callback))
        await asyncio.sleep(1)


async def speech_recognize(audio_info):
    '''识别语音为文本，接收语音数据audio-url参数，返回转译文本
    '''
    global COUNTER
    # while COUNTER > COUNTER_MAX:
    #     print(f'waiting in queque, cocurrency: {COUNTER}')
    #     await asyncio.sleep(1)
    try:
        COUNTER += 1
        Callback_param = {"task_id": audio_info.task_id}
        audio, msg = await asr_process.download(audio_info.file_path)
        if msg != 'ok':
            Callback_param['code'] = 4003
            Callback_param['msg'] = msg
            COUNTER -= 1
            return Callback_param
        if not audio:
            Callback_param['code'] = 4002
            Callback_param['msg'] = 'no audio data'
            COUNTER -= 1
            return Callback_param
        text, exception = await asr_process.rec(audio)
        if exception:
            Callback_param['code'] = 4004
            Callback_param['msg'] = f'{exception} times of getting exception'
        Callback_param['content'] = text
        COUNTER -= 1
    except Exception as e:
        logger.exception(e)
        Callback_param['code'] = 4000
        Callback_param['msg'] = str(e)
        COUNTER -= 1
    finally:
        # 回调接口调用
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(url=url, json=Callback_param) as resp:
                html = await resp.text() 
        if json.loads(html)["code"] == 1:
            # 识别完成，清理数据库
            config.url_db.Delete(audio_info.task_id.encode())
        else:
            logger("回调失败！！")