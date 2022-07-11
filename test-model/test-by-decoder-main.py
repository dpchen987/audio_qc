#!/usr/bin/env python3
# coding:utf-8


import os

IDR = 'Final result:'


def parse_log(logfile, save_to):
    with open(logfile) as f:
        lines = f.readlines()
    trans = []
    rtf = '0'
    for line in lines:
        if 'RTF' in line:
            print(line)
            rtf = line.split('RTF:')[-1].strip()
            continue
        if IDR not in line:
            continue
        a, b = line.split(IDR)
        uttid = a.strip().split()[-1]
        text = b.strip()
        trans.append(f'{uttid}\t{text}\n')
    with open(save_to, 'w') as f:
        f.write(''.join(trans))
    with open(f'{logfile}.rtf', 'w') as f:
        f.write(rtf)


def main(args):
    bin_dir = args.bin_dir
    model_dir = args.model_dir
    wav_scp = args.wav_scp
    trans = args.trans
    skip = args.skip
    # 1. gen result name prefix according args
    result_prefix = [
        bin_dir.strip('/').split('/')[-1],
        model_dir.strip('/').split('/')[-1],
        wav_scp.strip('/').split('/')[-1],
    ]
    result_prefix = '-'.join(result_prefix)

    os.environ['GLOG_logtostderr'] = '1'
    os.environ['GLOG_v'] = '2'
    # 2. check and set LD_LIBRARY_PATH
    names = os.listdir(bin_dir)
    for name in names:
        if name.endswith('.so') or '.so.' in name:
            os.environ['LD_LIBRARY_PATH'] = bin_dir
            break
    # 3. command for decoder_main
    onnx = False
    for name in names:
        if 'libonnxruntime.so' in name:
            onnx = True
            break
    cmds = [
        f'{bin_dir}/decoder_main --chunk_size -1 ',
        f'--wav_scp {wav_scp}',
        f'--dict_path {model_dir}/words.txt',
    ]
    if args.num_threads > 1:
        cmds.append(f'--num_threads {args.num_threads}')
    if onnx:
        print('test onnx model ...')
        cmds.append(f'--onnx_dir {model_dir}')
    else:
        cmds.append(f'--model_path {model_dir}/final.zip')
    cmds.append('2>&1')
    cmds.append(f'| tee {result_prefix}-log.txt')
    # 4. run decoder_main
    if not skip:
        cmd = ' '.join(cmds)
        print(cmd)
        os.system(cmd)
        os.system('echo ==========$LD_LIBRARY_PATH')
    else:
        print('skip running decoder_main')
    # 5. parse log
    asr_trans = f'{result_prefix}-asr-trans.txt'
    parse_log(f'{result_prefix}-log.txt', asr_trans)
    # 5. comput cer
    cmd = (f'python compute-wer.py --char=1 --v=1 '
           f'{trans} {asr_trans} > {result_prefix}.cer.txt')
    print(cmd)
    os.system(cmd)
    print('done')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='copy ldd of ELF')
    parser.add_argument('-b', '--bin_dir', required=True, help='dir of decoder_main')
    parser.add_argument('-m', '--model_dir', required=True, help='dir of model')
    parser.add_argument('-w', '--wav_scp', required=True, help='path to wav_scp file')
    parser.add_argument('-t', '--trans', required=True, help='path to trans file')
    parser.add_argument('-s', '--skip', default=False, help='skip running decoder_main')
    parser.add_argument('-n', '--num_threads', type=int, default=1, help='num of threads for ASR model')
    args = parser.parse_args()
    main(args)
