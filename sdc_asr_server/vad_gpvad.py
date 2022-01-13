from io import BytesIO
import soundfile as sf
from sdc_asr_server.gpvad.forward import GPVAD
from sdc_asr_server.logger import logger


VAD = GPVAD()


def cut(timeline, data, samplerate):
    for tl in timeline:
        start = int(tl[0] / 1000 * samplerate)
        end = int(tl[1] / 1000 * samplerate)
        segment = data[start: end]
        logger.info(f'segment: {start, end}')
        yield segment


def vad(audio):
    timeline = VAD.vad(BytesIO(audio))
    print('timeline:', timeline)
    data, samplerate = sf.read(BytesIO(audio), dtype='int16')
    duration = len(data) / samplerate
    segments = cut(timeline, data, samplerate)
    return segments, duration
