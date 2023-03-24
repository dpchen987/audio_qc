# encoding: utf8
import os
import asyncio
from fastapi import APIRouter, Body
from fastapi import Depends, Request, File, UploadFile
from asr_api_server import __version__
from asr_api_server import asr_process
from asr_api_server.logger import logger
from asr_api_server.data_model.api_model import ASRResponse, ASRHeaer, AudioInfo, RecognizeResponse, CallBackParam
from asr_api_server.asr_consumer import speech_recognize
from asr_api_server import config

def auth(appkey):
    return appkey == '123'


router = APIRouter()


@router.get('/status')
async def is_running():
    return {'version': __version__}


@router.post("/rec_file", response_model=ASRResponse)
async def recognize_file(
        query: ASRHeaer = Depends(),
        afile: UploadFile = File(...)):
    '''识别语音为文本，接收Form Data形式上传音频文件
    '''
    error = {
        'status': 4001,
        'message': 'Auth failed'
    }
    if not auth(query.appkey):
        return ASRResponse(**error)
    audio = await afile.read()
    if not audio:
        error['status'] = 4002
        error['message'] = 'no audio data'
        return ASRResponse(**error)
    result = await asr_process.rec(audio)
    response = {
        'taskid': '123',
        'result': result,
    }
    return ASRResponse(**response)


@router.post("/rec", response_model=ASRResponse)
async def recognize(request: Request, query: ASRHeaer = Depends()):
    '''识别语音为文本，接收语音数据有两种途径:
    1. 在header里面配置audio-url参数
    2. 以二进制流的方式放到http的body中，header设置：'Content-Type': 'application/octet-stream'
    '''
    error = {
        'status': 4001,
        'message': 'Auth failed'
    }
    if not auth(query.appkey):
        return ASRResponse(**error)
    if query.audio_url:
        audio, msg = await asr_process.download(query.audio_url)
        if msg != 'ok':
            error['status'] = 4003
            error['message'] = msg
            return ASRResponse(**error)
    else:
        audio = await request.body()
    if not audio:
        error['status'] = 4002
        error['message'] = 'no audio data'
        return ASRResponse(**error)
    text, exception = await asr_process.rec(audio)
    response = {
        'status': 2000,
        'message': 'success',
        'taskid': '123',
        'text': text,
        'exception': exception,
    }
    if exception:
        response['status'] = 4004
        response['message'] = f'{exception} times of getting exception'
    return ASRResponse(**response)


@router.post("/speech_rec", response_model=RecognizeResponse)
async def data_receive(audio_info: AudioInfo = Body(..., title="音频信息")):
    '''识别语音为文本，接收语音数据audio-url参数，返回转译文本
    '''
    response = {}
    if audio_info.task_id and audio_info.file_path:
        config.url_db.Put(audio_info.task_id.encode(), audio_info.file_path.encode())
        task = asyncio.create_task(speech_recognize(audio_info))
        # Add task to the set. This creates a strong reference.
        config.background_tasks.add(task)
        # To prevent keeping references to finished tasks forever,
        # make each task remove its own reference from the set after
        # completion:
        task.add_done_callback(config.background_tasks.discard)
        config.processing_set.add(audio_info.task_id)
    else:
        response['code'] = 4001
        response['msg'] = 'no task_id or file_path'
    return response

CALLBACK_COUNTER = 0

@router.post("/callBack_test", response_model=RecognizeResponse)
async def callback_test(callback_para: CallBackParam = Body(..., title="音频信息")):
    '''识别语音为文本，接收语音数据audio-url参数，返回转译文本
    '''
    global CALLBACK_COUNTER
    response = {}
    if callback_para.task_id:
        logger.info(f"{callback_para.task_id}:回调成功！")
        response['code'] = 0
        logger.info(f"{callback_para.err_msg=}")
        CALLBACK_COUNTER += 1
        print('============= ', f'{CALLBACK_COUNTER = }')
    return response
