import numpy as np

def myRfft(y: np.ndarray) -> np.ndarray:
    N = len(y)
    Np = N // 2 + 1
    res = np.zeros(Np, dtype=complex)
    for x in range(Np):
        for n in range(N):
            exp = np.exp(-1j*2*np.pi*x*n/N)
            res[x] += y[n] * exp
    return res