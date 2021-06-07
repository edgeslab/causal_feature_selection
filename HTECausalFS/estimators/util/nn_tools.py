import torch.nn as nn
from collections import OrderedDict


def simple_layer_creator(layer_dict=None,
                         layer_list=None,
                         n_inputs=1,
                         n_outputs=1):
    # Example of using Sequential with OrderedDict
    #     model = nn.Sequential(OrderedDict([
    #               ('conv1', nn.Conv2d(1,20,5)),
    #               ('relu1', nn.ReLU()),
    #               ('conv2', nn.Conv2d(20,64,5)),
    #               ('relu2', nn.ReLU())
    #             ]))
    if layer_dict is not None:
        od = OrderedDict()
        for layer in layer_dict:
            od[layer] = layer_dict[layer]
        model = nn.Sequential(od)
    elif layer_list is not None:
        od = OrderedDict()
        for i, layer in enumerate(layer_list):
            od[f"layer{i}"] = layer
        model = nn.Sequential(od)
    else:
        model = nn.Sequential(
            nn.Linear(n_inputs, 100),
            nn.Linear(100, n_outputs),
        )
    return model


def get_output_shape(model):
    out_features = 0
    for layer in model:
        if hasattr(layer, "out_features"):
            out_features = layer.out_features

    return out_features
