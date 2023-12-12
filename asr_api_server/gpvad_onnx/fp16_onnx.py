#!/usr/bin/env python3

import sys
path = sys.argv[1]
zz = path.rsplit('.', 1)
path_fp16 = zz[0] + '_fp16.' + zz[1]
print(path_fp16)

import onnx
from onnxconverter_common import float16

model = onnx.load(path)
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, path_fp16)
print('done')
