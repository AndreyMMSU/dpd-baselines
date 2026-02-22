import scipy
import torch
import numpy
from dpd_baselines.blocks import ComplexFIR


def make_BL(x, fs, fc, numtaps = 2*6):
    b = scipy.signal.firwin(numtaps, cutoff=fc,fs=fs, window="hann", pass_zero="lowpass")
    if isinstance(x, list):
        x = torch.tensor(x)
    if isinstance(x, numpy.ndarray):
        x = torch.from_numpy(x)
    b = torch.tensor(b, dtype=x.dtype).to(x.device)
    Fir = ComplexFIR(m = b.shape[0], coeff = b, trainable=False).to(x.device)

    return Fir(x), b 

    
        
