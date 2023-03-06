import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard
from onnx_tf.backend import prepare
import onnx


"""
converts torch training to ONNX model

:param model: torch model 
:param input_size: shape of model input
:param path: path.onnx

Source: https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model
"""


def convert_pth_onnx(model, input_size, path):
    # set the model to inference mode -- IMPORTANT!!
    model.eval()

    # dummy input tensor
    dummy_input = torch.randn(1, *input_size, requires_grad=True)

    # Export the model
    torch.onnx.export(model, dummy_input, path, export_params=True, opset_version=10, do_constant_folding=True)

    print(" ")
    print('Model has been converted to ONNX')


def convert_onnx_pb(model, input_size, path):
    onnx_model = onnx.load("./saved_models/Resnet18_CIFAR10.onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("./saved_models/Resnet18_CIFAR10.pb")