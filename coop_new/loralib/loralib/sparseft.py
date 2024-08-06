#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseFT(nn.Linear):
    # Simulates sparse finetuning of original model weights
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        type : int = 0,
        fan_in_fan_out : bool = False, 
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        self.type = type
        if (type == 0):
            self.mask_input = nn.Parameter(
                self.weight.new_zeros(in_features)
            )
            self.mask_output = nn.Parameter(
                self.weight.new_zeros(out_features)
            ) 
        elif (type == 1):
            self.mask_weight = nn.Parameter(
                self.weight.new_zeros((in_features, out_features))
            )
        else:
            raise NotImplementedError
        self.delta_weight = nn.Parameter(
            self.weight.new_zeros((in_features, out_features))
        )
        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if (self.type == 0):
            if hasattr(self, 'mask_input'):
                nn.init.constant_(self.mask_input, 1.0)
                nn.init.constant_(self.mask_output, 1.0)
                nn.init.zeros_(self.delta_weight)
        elif (self.type == 1):
            if hasattr(self, 'mask_weight'):
                nn.init.constant_(self.mask_weight, 1.0)
                nn.init.zeros_(self.delta_weight)
        else:
            raise NotImplementedError
        
    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), bias=self.bias)
        # if mask_input.requires grad has been set to False in run_glue, it mean we are running in discrete mode
        if (self.type == 0):
            if (self.mask_input.requires_grad):
                inp = x*torch.sigmoid(self.mask_input)
                out = inp @ self.delta_weight
                out = out*torch.sigmoid(self.mask_output)
                result += out
            else:
                inp = x[..., self.mask_input]
                out = inp @ self.delta_weight
                result[..., self.mask_output] += out
        elif (self.type == 1):
            delta_weight = torch.sigmoid(self.mask_weight)*self.delta_weight
            out = x @ delta_weight
            result += out
        else:
            raise NotImplementedError
        return result
            