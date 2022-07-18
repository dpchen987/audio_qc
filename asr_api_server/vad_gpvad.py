from io import BytesIO
import soundfile as sf
from asr_api_server.gpvad_onnx.infer_onnxruntime import GPVAD
from asr_api_server.logger import logger


VAD = GPVAD('sre')


def cut(timeline, data, samplerate):
    for tl in timeline:
        if tl[1] - tl[0] < 500:
            logger.info(f'too short audio segment {tl[1]-tl[0]}')
            continue
        start = int(tl[0] / 1000 * samplerate)
        end = int(tl[1] / 1000 * samplerate)
        segment = data[start: end]
        # logger.info(f'segment: {start, end}')
        yield segment


def vad(audio):
    bio = BytesIO(audio)
    timeline = VAD.vad(bio)
    bio.seek(0)
    data, samplerate = sf.read(bio, dtype='int16')
    duration = len(data) / samplerate
    segments = cut(timeline, data, samplerate)
    return segments, duration, samplerate


if __name__ == '__main__':
    import sys
    fp = sys.argv[1]
    data = open(fp, 'rb').read()
    s, d, sr = vad(data)
    print(s, d, sr)
