import torch
import torch.nn as nn
import numpy as np

class NoiseGenerator(nn.Module):
    def __init__(
            self):
        super().__init__()

    def forward(self, harmonic, noise_param,):

        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1

        return noise
