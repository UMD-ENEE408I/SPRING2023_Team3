import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy import signal

def normalize_data(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
