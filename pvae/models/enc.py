"""
plain encoder meta class
"""

import torch.nn as nn

class Enc(nn.Module):
    def __init__(self, enc, params):
        super(Enc, self).__init__()
        self.enc = enc
        self.modelName = None
        self.params = params
        self.data_size = params.data_size

    def getDataLoaders(self, batch_size, shuffle, device, *args):
        raise NotImplementedError

    def forward(self, x):
        encoder_result_raw = self.enc(x)
        encoder_result = encoder_result_raw[0]
        return encoder_result

