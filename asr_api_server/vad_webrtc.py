from io import BytesIO
import soundfile as sf
import collections
import time
import webrtcvad

from asr_api_server.logger import logger

g_vad = webrtcvad.Vad()


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                # yield b''.join([f.bytes for f in voiced_frames])
                yield voiced_frames
                ring_buffer.clear()
                voiced_frames = []
    # if triggered:
        # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    # sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        # yield b''.join([f.bytes for f in voiced_frames])
        yield voiced_frames


def vad(data, aggressiveness=3):
    ''' aggressiveness: 1, 2, 3'''
    bio = BytesIO(data)
    audio, samplerate = sf.read(bio, dtype='int16')
    duration = len(audio) / samplerate
    audio = audio.tobytes()
    g_vad.set_mode(aggressiveness)
    frame_duration_ms = 20
    frames = frame_generator(frame_duration_ms, audio, samplerate)
    segments = vad_collector(samplerate, frame_duration_ms, 300, g_vad, frames)
    seg_bytes = []
    for segment in segments:
        if len(segment) < 25:
            # 25 * 20ms = 500ms
            logger.info(f'too short segment {len(segment)*20}ms')
            continue
        seg_bytes.append(b''.join([f.bytes for f in segment]))
    return seg_bytes, duration, samplerate


if __name__ == '__main__':
    from sys import argv
    import sys
    import numpy as np
    if len(argv) != 2:
        print('Usage: example.py <path to wav file>\n')
        sys.exit(1)
    data = open(argv[1], 'rb').read()
    b = time.time()
    segments, duration, sr = vad(data)
    e = time.time()
    print(f'{len(segments) = }')
    for i, segment in enumerate(segments):
        path = 'chunk-%002d.wav' % (i,)
        print('Writing %s' % (path,))
        arr = np.frombuffer(segment, dtype='int16')
        sf.write(path, arr, sr)
    print('time:', e-b)
