import eelbrain as eel
import numpy as np
import scipy, pathlib, importlib, mne, time, os

def load_eeg(eegfile, ch='Cz', refs=['EXG1', 'EXG2']):
    '''
    load EDF, rereference and find Erg start time
    eegfile: path to .bdf file
    refc: list of reference channels
    '''

    print('LOADING', eegfile.stem)
    raw = mne.io.read_raw_bdf(eegfile)
    fs_eeg = raw.info['sfreq']

    erg = raw.get_data(picks=['Erg1'])
    erg_nd = eel.NDVar(erg[0,:], eel.UTS(0, 1/fs_eeg, erg.shape[1]))
    
    eeg = raw.get_data(picks=[ch])
    eeg_nd = eel.NDVar(eeg[0,:], eel.UTS(0, 1/fs_eeg,eeg.shape[1]))

    refs = raw.get_data(picks=refs)
    ref_nd = eel.NDVar(refs, (eel.Case, eel.UTS(0, 1/fs_eeg,eeg.shape[1]))).mean('case')

    eeg_reref = eeg_nd - ref_nd

    erg_start = np.where(erg_nd.x > 5*np.std(erg_nd.x))[0][0]+int(1/eeg_nd.time.tstep) # first peak in erg channel
    print('Erg1 start at ', erg_start/fs_eeg)
    
    eeg_a = eel.NDVar(eeg_reref.x[erg_start:], eel.UTS(0, eeg_reref.time.tstep, len(eeg_reref.x[erg_start:])))
    erg_a = eel.NDVar(erg_nd.x[erg_start:], eel.UTS(0, erg_nd.time.tstep, len(erg_nd.x[erg_start:])))

    return eeg_a, erg_a, erg_start

def load_eeg_multichannel(eegfile, channels, refs=['EXG1', 'EXG2'], sensordim='biosemi32', fsds=4096):
    '''
    load EDF, rereference and find Erg start time
    eegfile: path to .bdf file
    refc: list of reference channels
    '''
    if isinstance(sensordim, str):
        sensordim = eel.Sensor.from_montage(sensordim)

    print('LOADING', eegfile.stem)
    raw = mne.io.read_raw_bdf(eegfile)
    fs_eeg = raw.info['sfreq']

    erg = raw.get_data(picks=['Erg1'])
    erg_nd = eel.NDVar(erg[0,:], eel.UTS(0, 1/fs_eeg, erg.shape[1]))
    
    eeg = raw.get_data(picks=channels)
    eeg_nd = eel.NDVar(eeg, (sensordim, eel.UTS(0, 1/fs_eeg,eeg.shape[1])))

    refs = raw.get_data(picks=refs)
    ref_nd = eel.NDVar(refs, (eel.Case, eel.UTS(0, 1/fs_eeg,eeg.shape[1]))).mean('case')

    eeg_reref = eeg_nd - ref_nd

    erg_start = np.where(erg_nd.x > 5*np.std(erg_nd.x))[0][0]+int(1/eeg_nd.time.tstep) # first peak in erg channel
    print('Erg1 start at ', erg_start/fs_eeg)
    
    eeg_a = eel.NDVar(eeg_reref.x[:, erg_start:], (sensordim, eel.UTS(0, eeg_reref.time.tstep, len(eeg_reref.x[0, erg_start:]))))
    eeg_a = eel.resample(eeg_a, fsds)
    erg_a = eel.NDVar(erg_nd.x[erg_start:], eel.UTS(0, erg_nd.time.tstep, len(erg_nd.x[erg_start:])))
    erg_a = eel.resample(erg_a, fsds)
    return eeg_a, erg_a, erg_start


def preprocess_eeg_clicks(datadict, verbose=True, eegk='Cz'):
    eeg, erg = datadict[eegk], datadict['erg']
    erg -= erg.mean()
    ergp = erg.clip(min=0) # positive rectification
    ergn = -(erg.clip(max=0)) # negative rectification

    if verbose: print('filtering')
    eeg = eel.filter_data(eeg, 1, None)
    eeg.x = mne.filter.notch_filter(eeg.x, 1/eeg.time.tstep, [50*i for i in range(1, 20)], notch_widths=5)

    return eeg, erg, ergp, ergn


def preprocess_eeg_speech_preds(eegs, preds, verbose=True, tmin=1, tmax=289, filt_method='iir'):
    eegs = eel.combine([x.sub(time=(tmin, tmax)) for x in eegs if x.time.tmax>tmax])
    eegs = eel.NDVar(eegs.x, (eel.Case, eel.UTS(0, eegs.time.tstep, eegs.x.shape[1])))
    predlist = []
    for k in preds.keys():
        if isinstance(preds[k], list):
            for i in range(len(preds[k])):
                preds[k][i] = eel.combine([x.sub(time=(tmin, tmax)) for x in preds[k][i] if x.time.tmax>tmax])
                preds[k][i] = eel.NDVar(preds[k][i].x, (eel.Case, eel.UTS(0, preds[k][i].time.tstep, preds[k][i].x.shape[1])))
                predlist.append(preds[k][i])
        else:
            preds[k] = eel.combine([x.sub(time=(tmin, tmax)) for x in preds[k] if x.time.tmax>tmax])
            preds[k] = eel.NDVar(preds[k].x, (eel.Case, eel.UTS(0, preds[k].time.tstep, preds[k].x.shape[1])))
            predlist.append(preds[k])
            
    if filt_method == 'iir':
        fs=int(1/eegs.time.tstep)
        print('sosfilt fs', fs)
        filtsos = scipy.signal.butter(1, 1, btype='high', analog=False, output='sos', fs=fs)
        for i in range(len(eegs)):
            eegs.x[i,:] = scipy.signal.sosfilt(filtsos, eegs.x[i,:])
    elif filt_method == 'fir':
        eegs = eel.filter_data(eegs, 1, None)

    for i in range(len(eegs)):
        eegs.x[i,:] = mne.filter.notch_filter(eegs.x[i,:], 1/eegs.time.tstep, [50*i for i in range(1, 20)], notch_widths=5)

    if verbose: print('removing outliers')
    eegs2 = []
    predlist2 = []
    for i in range(len(predlist)):
        predlist2.append([])
    rNs = []
    for i in range(len(eegs)):
        yy = eel.filter_data(eegs[i], 1, 40)
        predlist1 = [x[i] for x in predlist]
        y2, predlist11, rN = remove_outliers(eegs[i], yy, predlist1, stdmul=5, verbose=verbose)
        eegs2.append(y2)
        for i in range(len(predlist11)):
            predlist2[i].append(predlist11[i])
        rNs.append(rN)

    predlist2 = [eel.combine(x) for x in predlist2]
    eegs2 = eel.combine(eegs2)

    preds2 = {}
    i = 0
    for k in preds.keys():
        if isinstance(preds[k], list):
            preds2[k] = []
            for ii in range(len(preds[k])):
                preds2[k].append(predlist2[i])
                i += 1
        else:
            preds2[k] = predlist2[i]
            i += 1

    return eegs2, preds2, rNs


def remove_outliers(y, y1, preds, zerotime=1, stdmul=5, verbose=True):
    '''
    removes outliers by setting 1s before and after high variance segments to zero
    y: eeg NDVar
    y1: filtered eeg NDVar to use for detecting outliers
    preds: list of predictor NDVars
    zerotime: time before and after outlier in seconds
    stdmul: threshold for detecting outliers
    '''
    stdval = y1.sub(time=(1, y1.time.tmax)).std()    
    idxs = np.where(y1.abs().x > stdmul*stdval)[0] # high variance indices
    y2 = y.copy() # for output
    preds2 = [pred.copy() for pred in preds] # for output 
    fs = int(1/y.time.tstep)
    rN = np.zeros_like(y2.x) # zeroed samples
    for i in idxs:
        i1 = np.max([0, i-zerotime*fs]) # 1s before artifact
        i2 = np.min([len(y2.x), i+zerotime*fs]) # 1s after artifact
        rN[i1:i2] = 1
        y2.x[i1:i2] = 0
        for pred2 in preds2:
            pred2.x[i1:i2] = 0
    if verbose: print(f'removed {np.sum(rN)} samples, {100*np.sum(rN)/len(rN)}%')
    return y2, preds2, 100*np.sum(rN)/len(rN)

def shift_NDVar(x, shift):
    newtime = eel.UTS(x.time.tmin+shift, x.time.tstep, len(x))
    return eel.NDVar(x.x, newtime, name=x.name)