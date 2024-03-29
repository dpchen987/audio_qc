import time
import os
import numpy as np
import librosa
import soundfile as sf
import sklearn.preprocessing as pre
import pickle
import onnxruntime

SAMPLE_RATE = 16000
# SAMPLE_RATE = 22050
EPS = np.spacing(1)
LMS_ARGS = {
    'n_fft': 2048,
    'n_mels': 64,
    'hop_length': int(SAMPLE_RATE * 0.02),
    'win_length': int(SAMPLE_RATE * 0.04)
}


def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters

    """

    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def double_threshold(x, high_thres, low_thres, n_connect=1):
    """double_threshold
    Helper function to calculate double threshold for n-dim arrays

    :param x: input array
    :param high_thres: high threshold value
    :param low_thres: Low threshold value
    :param n_connect: Distance of <= n clusters will be merged
    """
    assert x.ndim <= 3, "Whoops something went wrong with the input ({}), check if its <= 3 dims".format(
        x.shape)
    if x.ndim == 3:
        apply_dim = 1
    elif x.ndim < 3:
        apply_dim = 0
    else:
        apply_dim = 0
    # x is assumed to be 3d: (batch, time, dim)
    # Assumed to be 2d : (time, dim)
    # Assumed to be 1d : (time)
    # time axis is therefore at 1 for 3d and 0 for 2d (
    return np.apply_along_axis(lambda x: _double_threshold(
        x, high_thres, low_thres, n_connect=n_connect),
                               axis=apply_dim,
                               arr=x)


def connect_(pairs, n=1):
    """connect_
    Connects two adjacent clusters if their distance is <= n

    :param pairs: Clusters of iterateables e.g., [(1,5),(7,10)]
    :param n: distance between two clusters
    """
    if len(pairs) == 0:
        return []
    start_, end_ = pairs[0]
    new_pairs = []
    for i, (next_item, cur_item) in enumerate(zip(pairs[1:], pairs[0:])):
        end_ = next_item[1]
        if next_item[0] - cur_item[1] <= n:
            pass
        else:
            new_pairs.append((start_, cur_item[1]))
            start_ = next_item[0]
    new_pairs.append((start_, end_))
    return new_pairs


def _double_threshold(x, high_thres, low_thres, n_connect=1, return_arr=True):
    """_double_threshold
    Computes a double threshold over the input array

    :param x: input array, needs to be 1d
    :param high_thres: High threshold over the array
    :param low_thres: Low threshold over the array
    :param n_connect: Postprocessing, maximal distance between clusters to connect
    :param return_arr: By default this function returns the filtered indiced, but if return_arr = True it returns an array of tsame size as x filled with ones and zeros.
    """
    assert x.ndim == 1, "Input needs to be 1d"
    high_locations = np.where(x > high_thres)[0]
    locations = x > low_thres
    encoded_pairs = find_contiguous_regions(locations)

    filtered_list = list(
        filter(
            lambda pair:
            ((pair[0] <= high_locations) & (high_locations <= pair[1])).any(),
            encoded_pairs))

    filtered_list = connect_(filtered_list, n_connect)
    if return_arr:
        zero_one_arr = np.zeros_like(x, dtype=int)
        for sl in filtered_list:
            zero_one_arr[sl[0]:sl[1]] = 1
        return zero_one_arr
    return filtered_list


def decode_with_timestamps(encoder: pre.MultiLabelBinarizer, labels: np.array):
    """decode_with_timestamps
    Decodes the predicted label array (2d) into a list of
    [(Labelname, onset, offset), ...]

    :param encoder: Encoder during training
    :type encoder: pre.MultiLabelBinarizer
    :param labels: n-dim array
    :type labels: np.array
    """
    if labels.ndim == 3:
        return [_decode_with_timestamps(encoder, lab) for lab in labels]
    else:
        return _decode_with_timestamps(encoder, labels)


def _decode_with_timestamps(encoder, labels):
    result_labels = []
    for i, label_column in enumerate(labels.T):
        change_indices = find_contiguous_regions(label_column)
        # append [onset, offset] in the result list
        for row in change_indices:
            result_labels.append((encoder.classes_[i], row[0], row[1]))
    return result_labels


def extract_feature(wavefilepath):
    wav, sr = sf.read(wavefilepath, dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(-1)
    wav = librosa.resample(wav, sr, target_sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=wav.astype(np.float32), sr=SAMPLE_RATE, **LMS_ARGS)
    return np.log(mel + EPS).T


def extract_feature_mem(wav, sr):
    if wav.ndim > 1:
        wav = wav.mean(-1)
    if wav.dtype != 'float32':
        wav = wav.astype('float32')
    b = time.time()
    if sr != SAMPLE_RATE:
        print('-------- librosa.resample() ---')
        wav = librosa.resample(wav, sr, target_sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=wav.astype(np.float32), sr=SAMPLE_RATE, **LMS_ARGS)
    feat = np.log(mel + EPS).T
    print('---------- mel feature time', time.time() - b)
    return feat


class GPVAD:
    def __init__(self, model_name='t2bal', use_gpu=True, quant='') -> None:
        assert model_name in ['sre', 'a2_v2', 't2bal']
        self.quant = quant
        root_dir = os.path.dirname(os.path.abspath(__file__))
        if model_name == 'sre':
            model_path = os.path.join(root_dir, 'onnx_models/sre.onnx')
            if quant == 'int8':
                model_path = os.path.join(root_dir, 'onnx_models/sre_quant.onnx')
            elif quant == 'fp16':
                model_path = os.path.join(root_dir, 'onnx_models/sre_fp16.onnx')
        elif model_name == 't2bal':
            model_path = os.path.join(root_dir, 'onnx_models/t2bal.onnx')
            if quant == 'int8':
                model_path = os.path.join(root_dir, 'onnx_models/t2bal_quant.onnx')
            elif quant == 'fp16':
                model_path = os.path.join(root_dir, 'onnx_models/t2bal_fp16.onnx')
        else:
            model_path = os.path.join(root_dir, 'onnx_models/audio2_vox2.onnx')
            if quant == 'int8':
                model_path = os.path.join(root_dir, 'onnx_models/audio2_vox2_quant.onnx')
            elif quant == 'fp16':
                model_path = os.path.join(root_dir, 'onnx_models/audio2_vox2_fp16.onnx')
        if onnxruntime.get_device() == 'GPU' and use_gpu:
            # conf =  {
            #     "cudnn_conv_use_max_workspace": '1',
            #     "cudnn_conv1d_pad_to_nc1d": '1',
            #     # 'cudnn_conv_algo_search': 'EXHAUSTIVE',
            #     # 'do_copy_in_default_stream': True,
            # }
            # providers = [("CUDAExecutionProvider", conf)]
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ['CPUExecutionProvider']
        print(f'===== GPVAD {providers = }, {model_path = }')
        sess_opt = onnxruntime.SessionOptions()
        # sess_opt.intra_op_num_threads = 8
        self.model = onnxruntime.InferenceSession(model_path, sess_opt, providers=providers)
        self.model_resolution = 20  # miliseconds
        encoder_path = os.path.join(root_dir, 'labelencoders/vad.pkl')
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)
        self.threshold = (0.2, 0.01)  # 更好的recall，论文推荐(0.5, 0.1)
        self.speech_label_idx = np.where('Speech' == self.encoder.classes_)[0].squeeze()
        self.postprocessing_method = double_threshold

    def vad(self, audio_path):
        b = time.time()
        # wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE, res_type="soxr_hq")
        wav, sr = sf.read(audio_path, dtype='float32')
        print('======= read audio data time:', time.time() - b)
        b = time.time()
        ss = self.vad_mem(wav, sr)
        print('---------------- vad_mem() time ', time.time() - b, 'segments count:', len(ss))
        return ss
    
    def vad_tsk(self, wav, sr):
        # wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE, res_type="soxr_hq")
        b = time.time()
        ss = self.vad_mem(wav, sr)
        print('---------------- vad_mem() time ', time.time() - b, 'segments count:', len(ss))
        return ss

    def vad_mem(self, wav, sr):
        feature = extract_feature_mem(wav, sr)
        feature = np.expand_dims(feature, axis=0)
        if self.quant == 'fp16':
            feature = feature.astype('float16')
        output = []
        zz = self.model.run(None, {'modelInput': feature})
        prediction_tag, prediction_time = zz
        if prediction_time is not None:  # Some models do not predict timestamps
            thresholded_prediction = self.postprocessing_method(
                prediction_time, *self.threshold)
            labelled_predictions = decode_with_timestamps(
                self.encoder, thresholded_prediction)
            for label, start, end in labelled_predictions[0]:
                # print(label, start*self.model_resolution, end*self.model_resolution)
                if label != 'Speech': continue
                output.append([start*self.model_resolution, end*self.model_resolution])
        return output


if __name__ == "__main__":
    from sys import argv
    import time
    fn = argv[1]
    quant = False
    use_gpu = True
    results = []
    loop = 50
    for quant in ['int8', 'fp32']:
        pgvad = GPVAD('t2bal', use_gpu=use_gpu, quant=quant)
        times = []
        for i in range(loop):
            b = time.time()
            oo = pgvad.vad(fn)
            e = time.time() - b
            times.append(e)
            # print(i, 'time:', e, len(oo))
        avg = sum(times) / len(times)
        print(f'{quant = }, {avg = }')
        print('===' * 30)
        results.append((quant, avg))
    for r in results:
        print(f'{use_gpu = }, quant: {r[0]}, avg: {r[1]}')
    with open('z-vad-ts-onnx.txt', 'w') as f:
        ll = [f'{o[0]}\t{o[1]}\n' for o in oo]
        f.write(''.join(ll))
