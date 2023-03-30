
from typing import Optional
from fastapi import Header
from pydantic import BaseModel, Field


class ASRQuery:
    def __init__(self, appkey: str, format: Optional[str] = 'pcm'):
        self.appkey = appkey
        self.format = format


class ASRHeaer:
    def __init__(
        self,
        appkey: str = Header(None),
        format: Optional[str] = Header('pcm'),
        audio_url: Optional[str] = Header(''),
    ) -> None:
        self.appkey = appkey
        self.format = format
        self.audio_url = audio_url


class ASRResponse(BaseModel):
    task_id: str = Field('123', description='长度为32的任务ID', example='87f5401c9347beae7cc392c408217dd0')
    text: str = Field('', description='识别结果', example="北京的天气。")
    status: int = Field(2000, description='服务状态码', example=2000)
    message: str = Field('success', description='服务状态描述', example="success")
    exception: int = Field(0, description="ASR 解码过程中遇到的Exception次数", example=0)

class AudioInfo(BaseModel):
    task_id: str = Field(..., description='长度为32的任务ID', example='87f5401c9347beae7cc392c408217dd0')
    enable_punctution_prediction: bool = Field(True, description='是否在识别结果中增加标点')
    file_path: str = Field(..., description='文件路径', example="/data/audio/speech.pcm")
    callback_url: str = Field(..., description='回调地址', example="http://localhost:8305/asr/v1/callBack_test")
    # audio_format: str = Field('pcm', description="音频的格式", example="pcm/mp3")
    file_content: str = Field(' ', description='文件内容', example='8408217dd0')
    file_type: str = Field(..., description='文件类型', example="opus、wav")
    trans_type: int = Field(..., description='传输类型，1-文件路径，2-文件内容', example='2')

class CallBackParam(BaseModel):
    task_id: str = Field(..., description='长度为32的任务ID', example='87f5401c9347beae7cc392c408217dd0')
    code: int = Field(0, description='状态码: 0-成功, 异常码后续规定', example='0')
    content: str = Field(' ', description='识别结果', example="北京的天气。")
    err_msg: str = Field("success", description='返回状态的说明', example="success")
#     callback_path: str = Field(..., description='文件路径', example="/data/audio/speech.pcm")
    # audio_format: str = Field('pcm', description="音频的格式", example="pcm/mp3")

    
class RecognizeResponse(BaseModel):
    code: int = Field(0, description='状态码: 0-成功, 异常码后续规定', example='0')
    msg: str = Field("success", description='返回状态的说明', example="success")
    data: str = Field(' ', description='预留字段', example=" ")