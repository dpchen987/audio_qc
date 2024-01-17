import uvicorn

from fastapi import FastAPI, Body
from .vad_processor import speech_vad
from .api_model import AudioInfo, RecognizeResponse, CallBackParam
from .config import CONF
from .logger import logger
from .mqconn import mq


app = FastAPI(
    title="ASR Server",
)


@app.on_event('startup')
async def startup():
    await mq.init()


@app.post("/asr/v1/speech_rec", response_model=RecognizeResponse)
async def data_vad(audio_info: AudioInfo = Body(..., title="音频信息")):
    '''识别语音为文本，接收语音数据audio-url参数，返回转译文本
    '''
    response = {}
    # 传输类型判断
    if audio_info.trans_type == 1:
        # 传输类型: url
        response = await speech_vad(audio_info)
    elif audio_info.trans_type == 2:
        # 传输类型: bytes
        if audio_info.file_content and audio_info.file_type in ('wav', 'opus'):
            response = await speech_vad(audio_info)
        else:
            response['code'] = 4001
            response['msg'] = 'no file_content or wrong file_type'
    else:
        response['code'] = 4001
        response['msg'] = 'no task_id or file_path'
    return response


@app.post("/asr/v1/callBack_test", response_model=RecognizeResponse)
async def callback_test(data: CallBackParam = Body(..., title="音频信息")):
    '''识别语音为文本，接收语音数据audio-url参数，返回转译文本
    '''
    response = {}
    if data.task_id:
        logger.info(f"{repr(data.dict())}: 回调成功！")
        response['code'] = 0
    return response


def run_api():
    uvicorn.run(
        "asr_api_new.webapi:app",
        host=CONF['host'],
        port=CONF['port'],
        workers=CONF['workers'],
        loop='uvloop',
    )


if __name__ == '__main__':
    run_api()
