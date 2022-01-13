# encoding: utf8
from typing import Optional
from urllib import response
from fastapi import APIRouter
from fastapi import Depends, Body, Request, File, UploadFile
from sdc_asr_server import __version__
from sdc_asr_server import asr_process
from sdc_asr_server.data_model.api_model import ASRQuery, ASRResponse, ASRHeaer


def auth(appkey):
    return appkey=='123'


router = APIRouter()

@router.get('/status')
async def is_running():
    return {'version': __version__}


@router.post("/rec_file", response_model=ASRResponse)
async def recognize(query: ASRHeaer = Depends(), afile: UploadFile = File(...)):
    '''识别语音为文本，接收Form Data形式上传音频文件
    '''
    error = {
        'status': 4001,
        'message': 'Auth failed'
    }
    if not auth(query.appkey):
        return ASRResponse(**error)
    audio = await afile.read()
    print('recv audio data:', len(audio))
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
    print(request.headers)
    if not auth(query.appkey):
        return ASRResponse(**error)
    if query.audio_url:
        print('==== download', query.audio_url)
        audio, msg = await asr_process.download(query.audio_url)
        if msg != 'ok':
            error['status'] = 4003
            error['message'] = msg
            return ASRResponse(**error)
    else:
        print('==== get from octet-stream')
        audio = await request.body()
        print('recv audio data:', len(audio))
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


