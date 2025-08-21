import numpy as np
from abc import ABC, abstractmethod
from scipy.signal import freqz


class PhaseEstimator(ABC):
    """Abstract base class for phase estimation methods"""
    
    @abstractmethod
    def estimate_phase(self, data_window, **kwargs):
        """Estimate phase from data window"""
        pass


class ECHTEstimator(PhaseEstimator):
    """Endpoint-Correcting Hilbert Transform phase estimator"""
    
    def __init__(self, numerator, denominator, fs):
        self.numerator = numerator
        self.denominator = denominator
        self.fs = fs
    
    def _echt(self, xr, b, a, Fs, n=None):
        """
        Endpoint-correcting hilbert transform
        
        Parameters
        ----------
        xr: array like, input signal
        b: numerators of IIR filter response
        a: denominator of IIR filter response
        Fs: signal sampling rate
        n: length parameter
        
        Returns
        -------
        analytic signal
        """
        # Check input
        if n is None:
            n = len(xr)
        if not all(np.isreal(xr)):
            xr = np.real(xr)

        # Compute FFT
        x = np.fft.fft(xr, n)

        # Set negative components to zero and multiply positive by 2 (apart from DC and Nyquist frequency)
        h = np.zeros(n, dtype=x.dtype)
        if n > 0 and 2 * (n // 2) == n:
            # even and non-empty
            h[[0, n // 2]] = 1
            h[1:n // 2] = 2
        elif n > 0:
            # odd and non-empty
            h[0] = 1
            h[1:(n + 1) // 2] = 2
        x = x * h

        # Compute filter's frequency response
        T = 1 / Fs * n
        filt_freq = np.ceil(np.arange(-n/2, n/2)) / T
        filt_coeff = freqz(b, a, worN=filt_freq, fs=Fs)

        # Multiply FFT by filter's response function
        x = np.fft.fftshift(x)
        x = x * filt_coeff[1]
        x = np.fft.ifftshift(x)

        # IFFT
        x = np.fft.ifft(x)
        return x
    
    def estimate_phase(self, data_window, **kwargs):
        """Estimate phase using ecHT method"""
        analytic_signal = self._echt(
            data_window, 
            self.numerator, 
            self.denominator, 
            self.fs
        )
        return np.angle(analytic_signal)[-1] + np.pi
