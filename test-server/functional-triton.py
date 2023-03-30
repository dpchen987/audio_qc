import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np


TRITON_FLAGS = {
    'url': "localhost:8001",
    'verbose': False,
    'model_name': 'infer_pipeline',
}

def triton_rec(data: bytes) -> list:
    """
    测试triton
    :param data: int16字节音频数据
    :return:
    """
    results = []
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
    triton_client = grpcclient.InferenceServerClient(url=TRITON_FLAGS['url'], verbose=TRITON_FLAGS['verbose'])
    # with grpcclient.InferenceServerClient(
    #         url=TRITON_FLAGS['url'], verbose=TRITON_FLAGS['verbose']
    # ) as triton_client:
    response = triton_client.infer(
        TRITON_FLAGS['model_name'],
        inputs,
        request_id=str(sequence_id),
        outputs=outputs,
    )
    text = response.as_numpy("TRANSCRIPTS")[0].decode("utf-8")

    results.append(text)

    return results


if __name__ == '__main__':
    import soundfile as sf
    wav = '/aidata/audio/public/Leaderboard/datasets/SPEECHIO_ASR_ZH00006/wav/zv3V48f1X_A_0036.wav'
    waveform_arr, sr = sf.read(wav, dtype='int16')
    waveform_by = waveform_arr.tobytes()
    res = triton_rec(waveform_by)
    print(res)
