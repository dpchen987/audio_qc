import os
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


def load_joiner():
    dname = os.path.dirname(__file__)
    path = os.path.join(dname, 'gpvad_onnx/labelencoders/haha.wav')
    data, samplerate = sf.read(path, dtype='int16')
    duration = int(len(data) / samplerate * 1000)
    logger.info(f'haha wav {duration = }, {data.shape = }')
    return data, duration


JOINER_WAV, JOINER_LEN = load_joiner()


def cut_long(timeline, max_duration):
    new_timeline = []
    # print('cut_long....')
    for i, tl in enumerate(timeline):
        du = tl[1] - tl[0]
        # print(f'\t{i = }, {du = }')
        if du <= max_duration:  # or (i == len(timeline) - 1 and du < max_duration+2000):
            new_timeline.append(tl)
            continue
        j = 1
        # print(f'\t\tloooooooo: {i = }, {du = }')
        # 从头按max_duration切割，或者以下的均分切割
        count = du // max_duration
        if du % max_duration:
            count += 1
        step = du // count
        if du % count:
            step += 1
        while True:
            begin = tl[0] + (j-1) * step
            end = tl[0] + j * step
            x = [begin, end]
            new_timeline.append(x)
            du = tl[1] - end
            if du <= max_duration:
                x = [end, tl[1]]
                new_timeline.append(x)
                break
            j += 1
    return new_timeline


def cut(timeline, data, samplerate, max_duration=15000):
    new_timeline = cut_long(timeline, max_duration)
    segments = []
    last_duration = 0
    shorts = []
    # print('merging...')
    for i, tl in enumerate(new_timeline):
        duration = tl[1] - tl[0]
        # print(f'\t{i = }, {duration = }')
        start = int(tl[0] / 1000 * samplerate)
        end = int(tl[1] / 1000 * samplerate)
        segment = data[start: end]
        if ((duration + last_duration + JOINER_LEN <= max_duration) or
            (i == len(new_timeline) - 1 and duration < 2000)):
            # merge two segment to one
            # segments[-1] = np.concatenate((segments[-1], JOINER_WAV, segment))
            shorts.append(segment)
            shorts.append(JOINER_WAV)
            last_duration += duration + JOINER_LEN
        else:
            if len(shorts) == 2:
                segments.append(shorts[0])
            elif len(shorts) > 2:
                shorts.pop(-1)
                segments.append(np.concatenate(shorts))
            if duration >= max_duration:
                segments.append(segment)
                shorts = []
                last_duration = 0
            else:
                shorts = [segment, JOINER_WAV]
                last_duration = duration
    if len(shorts) == 2:
        segments.append(shorts[0])
    elif len(shorts) > 2:
        shorts.pop(-1)
        segments.append(np.concatenate(shorts))
            
    logger.info(f'===== vad: {len(timeline) = }, {len(new_timeline) = }, {len(segments) = }')
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
    data, samplerate = sf.read(bio, dtype='float32')
    timeline = VAD.vad_mem(data, samplerate)
    print(f'{timeline = }')
    data = (data * 32768.0).astype('int16')  # wav from float32 to int16
    duration = len(data) / samplerate
    # if vad_max < 10000ms, then 10000ms
    max_duration = max(CONF['vad_max'], 10000)
    segments = cut(timeline, data, samplerate, max_duration)
    return segments, duration, samplerate


if __name__ == '__main__':
    import sys
    fp = sys.argv[1]
    fname = fp.split('/')[-1]
    data = open(fp, 'rb').read()
    s, d, sr = vad(data)
    durations = [len(i)/sr for i in s]
    print(len(s), d, sr, f'{durations = }, {sum(durations) = }')
    for i, x in enumerate(s):
        sf.write(f'{fname}-{i}.wav', x, sr)
