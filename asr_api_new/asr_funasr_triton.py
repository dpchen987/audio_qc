import re
from . import config
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np

TRITON_FLAGS = {
    'url': "localhost:8001",
    'verbose': False,
    'model_name': 'infer_pipeline',
}


async def asr(data: bytes) -> str:
    """
    :param data: int16字节音频数据
    :return:
    """
    samples = np.frombuffer(data, dtype='int16')
    samples = np.array([samples], dtype=np.float32)
    lengths = np.array([[len(samples)]], dtype=np.int32)

    protocol_client = grpcclient
    inputs = [
        protocol_client.InferInput(
            "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
        ),
        protocol_client.InferInput(
            "WAV_LENS", lengths.shape, np_to_triton_dtype(lengths.dtype)
        ),
    ]
    inputs[0].set_data_from_numpy(samples)
    inputs[1].set_data_from_numpy(lengths)
    outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]
    sequence_id = 10086

    triton_client = grpcclient.InferenceServerClient(
        url=config.get_decoder_server_uri(),
        verbose=TRITON_FLAGS['verbose']
    )
    response = await triton_client.infer(
        TRITON_FLAGS['model_name'],
        inputs,
        request_id=str(sequence_id),
        outputs=outputs,
    )
    text = response.as_numpy("TRANSCRIPTS")[0].decode("utf-8")
    # print(f'origin {text = }')
    # 替换语音片段merge时加的哈哈（被识别为呵呵）
    text = re.sub(r'哈{2,8}|呵{2,8}|,{2,8}', ',', text)
    text = text.strip(',')
    # replace() is 9x faster than re.sub()
    # text = text.replace('哈哈', ',').replace('呵呵', ',')
    return text
