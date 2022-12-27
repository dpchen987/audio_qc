import time
import os
import torch
import torch.onnx
import onnx
from models import crnn


def convert(model_name):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if model_name == 'sre':
        model_path = os.path.join(root_dir, 'pretrained_models/sre/model.pth')
    elif model_name == 't2bal':
        model_path = os.path.join(root_dir, 'pretrained_models/t2bal/t2bal.pt')
    else:
        model_path = os.path.join(root_dir, 'pretrained_models/audio2_vox2/model.pth')
    model = crnn(
        outputdim=2,
        pretrained_from=model_path
    ).eval()
    dummy_input = torch.randn(1, 12211, 64, requires_grad=True)
    onnx_name = f'onnx_models/{model_name}.onnx'

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        onnx_name,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['modelInput'],  # the model's input names
        output_names = ['tag', 'time'], # the model's output names
        dynamic_axes={
            'modelInput' : {0: 'batch_size', 1: 'times'},    # variable length axes
            # 'tag': {0: 'batch_size'},
            # 'time': {0: 'batch_size', 1: 'times'},
        },
    )
    print(" ")
    print('Model has been converted to ONNX')
    onnxmodel = onnx.load(onnx_name)
    onnx.checker.check_model(onnxmodel)
    # print(onnx.helper.printable_graph(onnxmodel.graph))
    print('check_model done')


if __name__ == "__main__":
    convert('t2bal')
    convert('sre')
    convert('audio2_vox2')
