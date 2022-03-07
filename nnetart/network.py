# -*- coding: utf-8 -*-
"""
@title: random_art.py
@author: Tuan Le
@email: tuanle@hotmail.de

Implements a simple feed-forward neural network
"""

import torch
import torch.nn as nn

from typing import List


def weights_init_normal(m: nn.Module):
    """
    Change the standard deviations for the weight-attribute if you wanna experiment further.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1.0)
        m.bias.data.normal_(0.0, 0.1)
    return None


def init_activation_fnc(a):
    if a == "tanh":
        return nn.Tanh()
    elif a == "sigmoid":
        return nn.Sigmoid()
    elif a == "relu":
        return nn.ReLU()
    elif a == "softsign":
        return nn.Softsign()
    elif a == "sin":
        return torch.sin
    elif a == "cos":
        return torch.cos
    else:
        print(f"Inserted activation function {a} not compatible. Using tanh.")
        return nn.Tanh()


class FeedForwardNetwork(nn.Module):
    def __init__(self,
                 layers_dims: List = [10, 10, 10, 10, 10],
                 activation_fnc: str = "tanh",
                 colormode: str = "rgb",
                 alpha: bool = True):
        super(FeedForwardNetwork, self).__init__()
        colormode = colormode.lower()
        if colormode in ["rgb", "hsv", "hsl"]:
            if not alpha:
                out_nodes = 3
            else:
                out_nodes = 4
        elif colormode == "cmyk":
            if not alpha:
                out_nodes = 4
            else:
                out_nodes = 5
        elif colormode == "bw":
            if not alpha:
                out_nodes = 1
            else:
                out_nodes = 2
        else:
            print(f"wrong colormode {colormode} inserted in Neural Net. initialization.")
            raise ValueError
        input_layer = nn.Linear(in_features=5, out_features=layers_dims[0], bias=True)
        output_layer = nn.Linear(in_features=layers_dims[-1], out_features=out_nodes, bias=True)
        self.layers = nn.ModuleList([input_layer] +
                                    [nn.Linear(in_features=layers_dims[i],
                                               out_features=layers_dims[i+1], bias=True)
                                     for i in range(len(layers_dims)-1)] +
                                    [output_layer]
                                    )
        self.activation = init_activation_fnc(activation_fnc.lower())
        self.apply(weights_init_normal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, m in enumerate(self.layers):
            out = m(out)
            if i < len(self.layers)-1:
                out = self.activation(out)
            else:
                out = torch.sigmoid(out)
        return out