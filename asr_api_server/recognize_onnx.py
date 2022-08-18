import os
import yaml
import soundfile as sf
import onnxruntime as ort
import numpy as np

from swig_decoders import map_batch, \
    ctc_beam_search_decoder_batch, \
    TrieVector, PathTrie

try:
    from .easytimer import Timer
    from .fbank import calcFbank
    from .get_gpu_count import get_gpu_count
except:
    from easytimer import Timer
    from fbank import calcFbank


IGNORE_ID = -1


class Namespace:
    pass


class AsrOnnx:
    def __init__(self, model_dir, dtype='float32'):
        assert dtype in ['float16', 'float32']
        self.model_dir = model_dir
        args = Namespace()
        args.config = model_dir + 'train.yaml'
        args.gpu = -1
        args.dict = model_dir + 'words.txt'
        args.dtype = dtype
        args.mode = 'attention_rescoring'
        # args.mode = 'ctc_prefix_beam_search'
        self.args = args
        t = Timer()
        self.init_model()
        t.end('init_model')

    def init_model(self):
        gpus = get_gpu_count()
        print(f'========== Available GPU: {gpus} ==========')
        self.pid = os.getpid()
        self.ppid = os.getppid()
        if gpus > 0:
            self.args.gpu = (self.pid - self.ppid) % gpus
        else:
            self.args.gpu = -1
        args = self.args
        with open(args.config, 'r') as fin:
            self.config = yaml.load(fin, Loader=yaml.FullLoader)
        self.reverse_weight = self.config["model_conf"].get("reverse_weight", 0.0)
        print(f'{self.reverse_weight = }')

        # Init asr model from configs
        use_cuda = args.gpu >= 0 and ort.get_device() == 'GPU'
        if use_cuda:
            print('================ use_cuda...')
            print(f'--- {self.ppid=}, {self.pid=} use GPU={self.args.gpu} ---')
            EP_list = [
                    ('CUDAExecutionProvider', {'device_id': self.args.gpu}),
                    'CPUExecutionProvider']
            args.dtype = 'float16'
        else:
            print('================ use_cpu...')
            eps = ort.get_available_providers()
            if 'DnnlExecutionProvider' in eps:
                EP_list = ['DnnlExecutionProvider']
            elif 'OpenVINOExecutionProvider' in eps:
                EP_list = ['OpenVINOExecutionProvider']
            else:
                EP_list = ['CPUExecutionProvider']
            args.dtype = 'float32'
        print(EP_list)
        print(f'=== {args.dtype = } ===')
        if args.dtype == 'float16':
            encoder_onnx = 'encoder_fp16.onnx'
            decoder_onnx = 'decoder_fp16.onnx'
        else:
            encoder_onnx = 'encoder.onnx'
            decoder_onnx = 'decoder.onnx'
        args.encoder_onnx = os.path.join(self.model_dir, encoder_onnx)
        args.decoder_onnx = os.path.join(self.model_dir, decoder_onnx)

        so = ort.SessionOptions()
        # so.enable_profiling = True
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.encoder_ort = ort.InferenceSession(
                args.encoder_onnx,
                sess_options=so,
                providers=EP_list)
        self.decoder_ort = None
        if args.mode == "attention_rescoring":
            self.decoder_ort = ort.InferenceSession(
                    args.decoder_onnx,
                    sess_options=so,
                    providers=EP_list)

        # Load dict
        self.vocabulary = []
        char_dict = {}
        with open(args.dict, 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                char_dict[int(arr[1])] = arr[0]
                self.vocabulary.append(arr[0])
        self.eos = self.sos = len(char_dict) - 1

    def pad(self, feats):
        dim = 0
        max_len = max([f.shape[dim] for f in feats])
        new = []
        for feat in feats:
            if feat.shape[dim] < max_len:
                len_pad = max_len - feat.shape[dim]
                feat = np.pad(feat, ((0, len_pad), (0, 0)))
            new.append(feat)
        new = np.array(new)
        return new

    def calc_feat(self, wavs):
        fbank_conf = self.config['dataset_conf']['fbank_conf']
        t = Timer()
        feats = []
        feats_lengths = []
        t.begin('Fbank')
        for wav in wavs:
            mat = calcFbank(
                    wav,
                    filters_num=fbank_conf['num_mel_bins'],
                    win_length=fbank_conf['frame_length']/1000,
                    win_step=fbank_conf['frame_shift']/1000,
                    sample_rate=16000)
            feats.append(mat)
            feats_lengths.append(mat.shape[0])
        t.end()
        t.begin('padding')
        feats = self.pad(feats)
        feats_lengths = np.array(feats_lengths, dtype='int32')
        t.end()
        return feats, feats_lengths

    def rec(self, wavs):
        ''' batch recognize
        wavs: list of wav data
        '''
        eos, sos = self.eos, self.sos
        args = self.args
        feats, feats_lengths = self.calc_feat(wavs)
        feats = feats.astype(args.dtype)
        ort_inputs = {
                self.encoder_ort.get_inputs()[0].name: feats,
                self.encoder_ort.get_inputs()[1].name: feats_lengths}
        t = Timer()
        ort_outs = self.encoder_ort.run(None, ort_inputs)
        t.end('encoder_ort.run()')
        (encoder_out, encoder_out_lens, ctc_log_probs,
         beam_log_probs, beam_log_probs_idx) = ort_outs
        beam_size = beam_log_probs.shape[-1]
        batch_size = beam_log_probs.shape[0]
        num_processes = min(os.cpu_count(), batch_size)
        t.begin('decding...')
        if args.mode == 'ctc_greedy_search':
            if beam_size != 1:
                log_probs_idx = beam_log_probs_idx[:, :, 0]
            batch_sents = []
            for idx, seq in enumerate(log_probs_idx):
                batch_sents.append(seq[0:encoder_out_lens[idx]].tolist())
            hyps = map_batch(batch_sents, self.vocabulary, num_processes,
                             True, 0)
        elif args.mode in ('ctc_prefix_beam_search', "attention_rescoring"):
            batch_log_probs_seq_list = beam_log_probs.tolist()
            batch_log_probs_idx_list = beam_log_probs_idx.tolist()
            batch_len_list = encoder_out_lens.tolist()
            batch_log_probs_seq = []
            batch_log_probs_ids = []
            batch_start = []  # only effective in streaming deployment
            batch_root = TrieVector()
            root_dict = {}
            for i in range(len(batch_len_list)):
                num_sent = batch_len_list[i]
                batch_log_probs_seq.append(
                    batch_log_probs_seq_list[i][0:num_sent])
                batch_log_probs_ids.append(
                    batch_log_probs_idx_list[i][0:num_sent])
                root_dict[i] = PathTrie()
                batch_root.append(root_dict[i])
                batch_start.append(True)
            score_hyps = ctc_beam_search_decoder_batch(batch_log_probs_seq,
                                                       batch_log_probs_ids,
                                                       batch_root,
                                                       batch_start,
                                                       beam_size,
                                                       num_processes,
                                                       0, -2, 0.99999)
            if args.mode == 'ctc_prefix_beam_search':
                hyps = []
                for cand_hyps in score_hyps:
                    hyps.append(cand_hyps[0][1])
                hyps = map_batch(hyps, self.vocabulary, num_processes, False, 0)
        if args.mode == 'attention_rescoring':
            ctc_score, all_hyps = [], []
            max_len = 0
            for hyps in score_hyps:
                cur_len = len(hyps)
                if len(hyps) < beam_size:
                    hyps += (beam_size - cur_len) * [(-float("INF"), (0,))]
                cur_ctc_score = []
                for hyp in hyps:
                    cur_ctc_score.append(hyp[0])
                    all_hyps.append(list(hyp[1]))
                    if len(hyp[1]) > max_len:
                        max_len = len(hyp[1])
                ctc_score.append(cur_ctc_score)
            ctc_score = np.array(ctc_score, dtype=args.dtype)
            hyps_pad_sos_eos = np.ones(
                (batch_size, beam_size, max_len + 2),
                dtype=np.int64) * IGNORE_ID
            r_hyps_pad_sos_eos = np.ones(
                (batch_size, beam_size, max_len + 2),
                dtype=np.int64) * IGNORE_ID
            hyps_lens_sos = np.ones((batch_size, beam_size), dtype=np.int32)
            k = 0
            for i in range(batch_size):
                for j in range(beam_size):
                    cand = all_hyps[k]
                    l = len(cand) + 2
                    hyps_pad_sos_eos[i][j][0:l] = [sos] + cand + [eos]
                    r_hyps_pad_sos_eos[i][j][0:l] = [sos] + cand[::-1] + [eos]
                    hyps_lens_sos[i][j] = len(cand) + 1
                    k += 1
            decoder_ort_inputs = {
                self.decoder_ort.get_inputs()[0].name: encoder_out,
                self.decoder_ort.get_inputs()[1].name: encoder_out_lens,
                self.decoder_ort.get_inputs()[2].name: hyps_pad_sos_eos,
                self.decoder_ort.get_inputs()[3].name: hyps_lens_sos,
                self.decoder_ort.get_inputs()[-1].name: ctc_score}
            if self.reverse_weight > 0:
                r_hyps_pad_sos_eos_name = self.decoder_ort.get_inputs()[4].name
                decoder_ort_inputs[r_hyps_pad_sos_eos_name] = r_hyps_pad_sos_eos
            best_index = self.decoder_ort.run(None, decoder_ort_inputs)[0]
            best_sents = []
            k = 0
            for idx in best_index:
                cur_best_sent = all_hyps[k: k + beam_size][idx]
                best_sents.append(cur_best_sent)
                k += beam_size
            hyps = map_batch(best_sents, self.vocabulary, num_processes)
        t.end('decoder')
        return hyps


if __name__ == '__main__':
    wavfiles = [
        '/aidata/audio/public/Leaderboard/datasets/SPEECHIO_ASR_ZH00000/wav/Mo78NorEv-A_0001.wav',
        '/aidata/audio/public/Leaderboard/datasets/SPEECHIO_ASR_ZH00000/wav/Mo78NorEv-A_0002.wav',
        '/aidata/audio/public/Leaderboard/datasets/SPEECHIO_ASR_ZH00000/wav/Mo78NorEv-A_0003.wav',
        '/aidata/audio/public/Leaderboard/datasets/SPEECHIO_ASR_ZH00000/wav/Mo78NorEv-A_0004.wav',
        '/aidata/audio/public/Leaderboard/datasets/SPEECHIO_ASR_ZH00000/wav/Mo78NorEv-A_0005.wav',
        '/aidata/audio/public/Leaderboard/datasets/SPEECHIO_ASR_ZH00000/wav/Mo78NorEv-A_0006.wav',
        '/aidata/audio/public/Leaderboard/datasets/SPEECHIO_ASR_ZH00000/wav/Mo78NorEv-A_0007.wav',
        '/aidata/audio/public/Leaderboard/datasets/SPEECHIO_ASR_ZH00000/wav/Mo78NorEv-A_0008.wav',
    ]
    wavs = []
    for wavf in wavfiles:
        data, sr = sf.read(wavf, dtype='int16')
        assert sr == 16000
        print(f'{type(data)}, {data.shape}, data.dtype')
        wavs.append(data)
    model_dir = '/aidata/wenet-bin/20220506_u2pp_conformer_onnx_gpu/'
    asr = AsrOnnx(model_dir)
    texts = asr.rec(wavs)
    print(texts)
    input('>')
