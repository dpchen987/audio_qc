from io import BytesIO
import soundfile as sf
import numpy as np
from asr_api_server.gpvad_onnx.infer_onnxruntime import GPVAD
from asr_api_server.logger import logger

# 可选模型：'sre', 'a2_v2', 't2bal', (default:'t2bal').
VAD = GPVAD()


def cut(timeline, data, samplerate):
    segments = []
    last_duration = 0
    max_duration = 3000  # ms
    for i, tl in enumerate(timeline):
        duration = tl[1] - tl[0]
        start = int(tl[0] / 1000 * samplerate)
        end = int(tl[1] / 1000 * samplerate)
        segment = data[start: end]
        if i == 0:
            segments.append(segment)
            last_duration = duration
            continue
        if duration < max_duration and (
                i == len(timeline) - 1 or
                last_duration < max_duration):
            # this seg append to last seg
            segments[-1] = np.append(segments[-1], segment)
            last_duration += duration
        else:
            segments.append(segment)
            last_duration = duration
    print(f'===== vad: {len(timeline) = }, {len(segments) = }')
    return segments


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
    for i, x in enumerate(s):
        sf.write(f'{i}.wav', x, sr)
