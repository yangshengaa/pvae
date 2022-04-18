# Base Autoencoder Class 

import torch.nn as nn

class AE(nn.Module):
    def __init__(self, enc, dec, params):
        super(AE, self).__init__()
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params
        self.data_size = params.data_size

    def getDataLoaders(self, batch_size, shuffle, device, *args):
        raise NotImplementedError

    def forward(self, x):
        encoder_result_raw = self.enc(x)
        encoder_result = encoder_result_raw[0]
        decoder_result_raw = self.dec(encoder_result)
        decoder_result = decoder_result_raw[0]
        return decoder_result

