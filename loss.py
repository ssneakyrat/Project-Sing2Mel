import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
import numpy as np

class HybridLoss(nn.Module):
    def __init__(self, n_ffts, n_affts):
        super().__init__()
        self.loss_mss_func = MSSLoss(n_ffts)
        self.loss_amss_func = MSSLoss(n_affts)

    def forward(self, y_pred, y_true):
        loss_mss = self.loss_mss_func(y_pred, y_true)
        loss_amss = self.loss_amss_func(y_pred, y_true)

        loss = loss_mss * 0.7 + loss_amss * 0.3
        
        return loss, loss_mss, loss_amss
    
class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, overlap=0.75, eps=1e-7, name='SSSLoss'):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)
        self.name = name
    def forward(self, x_true, x_pred):
        min_len = np.min([x_true.shape[1], x_pred.shape[1]])
        
        # print('--------')
        # print(min_len)
        # print('x_pred:', x_pred.shape)
        # print('x_true:', x_true.shape)

        x_true = x_true[:, -min_len:]
        x_pred = x_pred[:, -min_len:]

        # print('x_pred:', x_pred.shape)
        # print('x_true:', x_true.shape)
        # print('--------\n\n\n')

        S_true = self.spec(x_true)
        S_pred = self.spec(x_pred)
        linear_term = F.l1_loss(S_pred, S_true)
        log_term = F.l1_loss((S_true + self.eps).log2(), (S_pred + self.eps).log2())

        loss = linear_term + self.alpha * log_term
        return {'loss':loss}
    
class MSSLoss(nn.Module):
    """
    Multi-scale Spectral Loss.
    Usage ::
    mssloss = MSSLoss([2048, 1024, 512, 256], alpha=1.0, overlap=0.75)
    mssloss(y_pred, y_gt)
    input(y_pred, y_gt) : two of torch.tensor w/ shape(batch, 1d-wave)
    output(loss) : torch.tensor(scalar)

    48k: n_ffts=[2048, 1024, 512, 256]
    24k: n_ffts=[1024, 512, 256, 128]
    """

    def __init__(self, n_ffts, alpha=1.0, ratio = 1.0, overlap=0.75, eps=1e-7, use_reverb=True, name='MultiScaleLoss'):
        super().__init__()
        self.losses = nn.ModuleList([SSSLoss(n_fft, alpha, overlap, eps) for n_fft in n_ffts])
        self.ratio = ratio
        self.name = name
    def forward(self, x_pred, x_true, return_spectrogram=True):
        x_pred = x_pred[..., :x_true.shape[-1]]
        if return_spectrogram:
            losses = []
            spec_true = []
            spec_pred = []
            for loss in self.losses:
                loss_dict = loss(x_true, x_pred)
                losses += [loss_dict['loss']]
        
        return self.ratio*sum(losses).sum()