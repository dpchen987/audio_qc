
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
