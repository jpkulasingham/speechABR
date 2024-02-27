
import eelbrain as eel
import numpy as np
import scipy, pathlib, importlib, mne, time, os
from pyfftw.interfaces.scipy_fftpack import fft as fftw
from pyfftw.interfaces.scipy_fftpack import ifft as ifftw

def fit_ERP(y, x, t1=-0.01, t2=0.03, stdmul=5, gap=0.004, N=-1, verbose=True, weighted=True, divbystd=False):
    '''
    extract ERP
    y: eeg NDVar
    x: click onsets NDVar
    t1: tstart ERP
    t2: tend ERP
    stdmul: threshold to detect onsets
    gap: gap between onsets (seconds)
    '''
    fs = 1/y.time.tstep
    trigs = np.where(x.abs().x > stdmul*x.std())[0] # find onset indices
    triggers = trigs[:-1][np.diff(trigs) > gap*fs] # remove indices that are too close together
    
    # placeholder for ERP
    erp = y.sub(time=(1+t1, 1+t2)).copy()
    erp = eel.NDVar(erp.x, eel.UTS(t1, y.time.tstep, len(erp)))
    erp.x = np.zeros(len(erp.x))
    # compute ERP
   # print(len(triggers))
    if N!=-1:
        triggers = triggers[:N]
    if verbose: print('num triggers =', len(triggers))
    for i, trig in enumerate(triggers):
        tt = x.time.times[trig]
        dd =  y.sub(time=(tt+t1, tt+t2)).x
        if divbystd: dd /= np.std(dd)
        erp.x += dd
    erp.x /= len(triggers)

    erp = eel.NDVar(erp.x, eel.UTS(t1, erp.time.tstep, len(erp)), name='ERP').sub(time=(t1, t2))
    
    return erp, triggers


def fit_freq_trf(y, x, t1=-5, t2=5):
    '''
    fits TRFs in the frequency domain
    based on Polonenko github code
    y: eeg NDVar
    x: predictor NDVar
    reg: regularization 
    '''
    if len(y.shape) == 2:
        t_weights = 1.0 / np.var(y.x, axis=-1, keepdims=True)
        t_weights /= t_weights.sum(0, keepdims=True)
        fft_x = fftw(x.x)
        fft_y = fftw(y.x)
        den = (np.conj(fft_x) * fft_x).mean(0)
        w = ifftw(
            (np.conj(fft_x) * fft_y * t_weights).sum(0)
            / den
        ).real
    else:
        fft_x = fftw(x.x)
        fft_y = fftw(y.x)
        den = np.conj(fft_x) * fft_x
        w = ifftw(
            (np.conj(fft_x) * fft_y)
            / den
        ).real

    # as reported in the Polonenko paper, 
    # w has lags from [0 to T/2] and [-T/2 to 0] where T is the total time of the data
    # reorder samples of w to have lags from [-T/2 to T/2]
    wNh = int(len(w)/2)
    xx = np.hstack([w[-wNh:], w[:wNh]])
    if len(w)!= len(xx):
        wlen = len(w)-1
    else:
        wlen = len(w)

    trf = eel.NDVar(xx, eel.UTS(-y.time.tmax/2, y.time.tstep, wlen))

    # only need a short segment of trf around 0
    trf = trf.sub(time=(t1, t2))
    return trf


def fit_trf_posneg(eeg, xp, xn, trfstr='', t1=-5, t2=5):
    trfp = fit_freq_trf(eeg, xp, t1=t1, t2=t2) # positive TRF
    trfp.name = 'pos TRF'+trfstr
    trfn = fit_freq_trf(eeg, xn, t1=t1, t2=t2) # negative TRF
    trfn.name = 'neg TRF'+trfstr
    trf = 0.5*(trfp+trfn) # average TRF
    trf.name = 'avg TRF'+trfstr
    return trf, trfp, trfn



def fit_trf_posonly(eeg, xp, trfstr=''):
    trf = fit_freq_trf(eeg, xp) # positive TRF
    trf.name = 'TRF'+trfstr
    return trf

