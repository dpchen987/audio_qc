from io import BytesIO
import soundfile as sf
import numpy as np
from .logger import logger
from .config import CONF
import librosa

if CONF['vad_frame'] == 'onnxruntime':
    logger.info('===== VAD framework : onnxruntime', )
    from asr_api_server.gpvad_onnx.infer_onnxruntime import GPVAD
else:
    logger.info('===== VAD framework : pytorch', )
    from asr_api_server.gpvad.forward import GPVAD

# 可选模型：'sre', 'a2_v2', 't2bal', (default:'t2bal').
VAD = GPVAD(use_gpu=CONF['vad_gpu'])
SAMPLE_RATE = 16000


def cut_timeline(timeline, max_duration):
    print(f'old timeline: {timeline}')
    new_timeline = []
    for i, tl in enumerate(timeline):
        du = tl[1] - tl[0]
        if du <= max_duration or (i == len(timeline) - 1 and du < 20):
            new_timeline.append(tl)
            continue
        j = 1
        while True:
            x = [tl[0] + (j-1) * max_duration, tl[0] + j * max_duration]
            new_timeline.append(x)
            du = tl[1] - j * max_duration
            if du <= max_duration:
                x = [tl[0] + j * max_duration, tl[1]]
                new_timeline.append(x)
                break
            j += 1
    print(f'new timeline: {new_timeline}')
    return new_timeline


def cut(timeline, data, samplerate, max_duration=15000):
    new_timeline = cut_timeline(timeline, max_duration)
    segments = []
    last_duration = 0
    for i, tl in enumerate(new_timeline):
        duration = tl[1] - tl[0]
        start = int(tl[0] / 1000 * samplerate)
        end = int(tl[1] / 1000 * samplerate)
        segment = data[start: end]
        if i == 0:
            segments.append(segment)
            last_duration = duration
            continue
        if duration + last_duration <= max_duration:
        # merge two segment to one
            segments[-1] = np.append(segments[-1], segment)
            last_duration += duration
        else:
            segments.append(segment)
            last_duration = duration
    print(f'===== vad: {len(timeline) = }, {len(new_timeline) = }, {len(segments) = }')
    duration_old = sum([t[1] - t[0] for t in timeline])
    duration_new = sum([t[1] - t[0] for t in new_timeline])
    print(f'{duration_old = }, {duration_new = }')
    return segments


def vad_duration(audio, prevad=False):
    "vad_duration"
    bio = BytesIO(audio)
    if prevad:
        wav, sr = sf.read(bio, stop=SAMPLE_RATE * 10, dtype='float32')
    else:
        wav, sr = sf.read(bio, start=SAMPLE_RATE * 8, dtype='float32')
        logger.info(f'----Full vad !----')
    if not wav.size:
        logger.info(f'empty audio !')
        return 0
    if sr != SAMPLE_RATE:
        wav, sr = librosa.load(wav, sr=SAMPLE_RATE, res_type="soxr_hq")
    timeline = VAD.vad_tsk(wav, sr)
    total = 0
    for i, tl in enumerate(timeline):
        duration = tl[1] - tl[0]
        total = max(duration, total)
    total = total / 1000
    return total


def vad(audio):
    bio = BytesIO(audio)
    timeline = VAD.vad(bio)
    bio.seek(0)
    data, samplerate = sf.read(bio, dtype='int16')
    duration = len(data) / samplerate
    max_duration = max(CONF['vad_max'], 10000)  # if vad_max < 10000, then 10000
    segments = cut(timeline, data, samplerate)
    return segments, duration, samplerate


if __name__ == '__main__':
    import sys
    fp = sys.argv[1]
    data = open(fp, 'rb').read()
    s, d, sr = vad(data)
    durations = [len(i)/sr for i in s]
    print(len(s), d, sr, f'{durations = }, {sum(durations) = }')
    for i, x in enumerate(s):
        sf.write(f'{i}.wav', x, sr)
