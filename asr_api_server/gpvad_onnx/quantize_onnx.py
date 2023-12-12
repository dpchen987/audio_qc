#!/usr/bin/env python3

import sys

model_fp32 = sys.argv[1]
zz = model_fp32.rsplit('.', 1)
model_quant = zz[0] + '_quant.' + zz[1]

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
print('done')
