# encoding: utf8
from fastapi import APIRouter
from fastapi import Depends, Request, File, UploadFile
from asr_api_server import __version__
from asr_api_server import asr_process
from asr_api_server.logger import logger
from asr_api_server.data_model.api_model import ASRResponse, ASRHeaer


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
    print(response)
    return ASRResponse(**response)
