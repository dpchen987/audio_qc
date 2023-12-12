# Visit http://localhost:8000/docs (default) for documents.

import time
import uuid
import json
import uvicorn
from typing import Annotated, Union

from fastapi import FastAPI, Response, Body
from .vad_processor import speech_vad
from .data_model.api_model import AudioInfo, RecognizeResponse
from .config import CONF
from .logger import logger


app = FastAPI(
    title="VAD Server",
    # openapi_url="/openapi.json",
    # root_path='/asr/v1',
    # root_path_in_servers=False,
)


@app.post("/asr/v1/speech_vad", response_model=RecognizeResponse)
async def data_vad(audio_info: AudioInfo = Body(..., title="音频信息")):
    '''识别语音为文本，接收语音数据audio-url参数，返回转译文本
    '''
    response = {}
    # 传输类型判断
    logger.info(f"task: {audio_info.task_id} enter vad api !-!")
    if audio_info.trans_type == 1 and audio_info.task_id and audio_info.file_path:
        # 传输类型: url
        response = await speech_vad(audio_info)
        # Add task to the set. This creates a strong reference.
        # To prevent keeping references to finished tasks forever,
        # make each task remove its own reference from the set after completion:
    elif audio_info.trans_type == 2 and audio_info.task_id and audio_info.file_path:
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


def run_api():
    uvicorn.run(
        "__main__:app",
        host=CONF['host'],
        port=CONF['port'],
        workers=CONF['vad_workers'],
        loop='uvloop',
    )


if __name__ == '__main__':
    run_api()
