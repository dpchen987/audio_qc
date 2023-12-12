from io import BytesIO
import soundfile as sf
import numpy as np
from .logger import logger
from .config import CONF
if CONF['vad_frame'] == 'onnxruntime':
    logger.info('===== VAD framework : onnxruntime', )
    from asr_api_server.gpvad_onnx.infer_onnxruntime import GPVAD
else:
    logger.info('===== VAD framework : pytorch', )
    from asr_api_server.gpvad.forward import GPVAD

# 可选模型：'sre', 'a2_v2', 't2bal', (default:'t2bal').
VAD = GPVAD('t2bal', use_gpu=CONF['vad_gpu'])


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


def cut_to_max(segments, samplerate):
    vad_max = CONF['vad_max']
    max_len = int(vad_max * samplerate)
    new = []
    for s in segments:
        if len(s) < max_len:
            new.append(s)
            continue
        count = len(s) // max_len
        if len(s) % max_len:
            count += 1
        step = len(s) // count
        if len(s) % count:
            step += 1
        for i in range(count):
            begin = i * step
            end = begin + step
            n = s[begin:end]
            new.append(n)
    return new


def vad_duration(audio):
    bio = BytesIO(audio)
    timeline = VAD.vad(bio)
    total = 0
    for i, tl in enumerate(timeline):
        duration = tl[1] - tl[0]
        total += duration
    total = total / 1000
    return total


def vad(audio):
    bio = BytesIO(audio)
    timeline = VAD.vad(bio)
    bio.seek(0)
    data, samplerate = sf.read(bio, dtype='int16')
    duration = len(data) / samplerate
    segments = cut(timeline, data, samplerate)
    if CONF['vad_max']:
        logger.info("-- before cut_to_max(), {len(segments) = }, {CONF['vad_max'] = }")
        segments = cut_to_max(segments, samplerate)
        logger.info("== after cut_to_max(), {len(segments) = }, {CONF['vad_max'] = }")
    return segments, duration, samplerate


if __name__ == '__main__':
    import sys
    fp = sys.argv[1]
    data = open(fp, 'rb').read()
    s, d, sr = vad(data)
    print(len(s), d, sr)
    for i, x in enumerate(s):
        sf.write(f'{i}.wav', x, sr)
