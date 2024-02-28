import eelbrain as eel
import numpy as np
import scipy, pathlib, importlib, mne, time, os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import scipy.stats
from tqdm import tqdm
from matplotlib.lines import Line2D


def plot_fig_AVG3(resdict, corrdict, savepath, sigbars=None):
    colorsA =[(0.5, 0.5, 1), (1, 0.5, 0.5), (0.2, 0.2, 0.6), (0.6, 0.2, 0.2),]
    ylim = [-0.5, 0.8]
    savepath.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(8, 15))
    plt.subplot(3,1,1)
    plot_avg(resdict, ['io_click_erp', 'in_click_erp', 's_click_erp'], labels=['Control', 'Inserts', 'Sound-field'], title='Click ERP', fig=fig, savefile=None,
                            colors=[(0.5,0.5,0.5), colorsA[2], colorsA[3]], ylim=ylim, y_sigbar=ylim[0]*1.1)
    plt.ylabel('Amplitude [uV]')
    ym = resdict['s_click_erp'].mean('case')
    plt.plot(ym.time.times*1e3+4.3, ym.x, color=colorsA[3], alpha=0.8, lw=2, linestyle='dotted')
    plt.xlim([-10, 30])


    plt.subplot(3,1,2)
    plot_avg(resdict, ['in_speech_ANnull', 'in_speech_RS', 's_speech_RS', 'in_speech_AN', 's_speech_AN'], labels=['Null model', 'Inserts RS', 'Sound-field RS', 'Inserts ANM', 'Sound-field ANM'], title='Speech TRF', fig=fig, savefile=None,
                            colors=[(0.5,0.5,0.5)]+colorsA, ylim=ylim, y_sigbar=ylim[0]*1.1)
    plt.ylabel('Amplitude [a.u.]')
    plt.xlim([-10, 30])

    plt.subplot(3,1,3)
    colors = [(0.5, 0.5, 1, 0.6), (1, 0.5, 0.5, 0.6),  (0.4, 0.4, 1), (1, 0.4, 0.4),]    
    plot_corr(plt.gca(), ['in_speech_RS', 's_speech_RS', 'in_speech_AN', 's_speech_AN'], 
                        ['RS inserts', 'RS sound-field', 'ANM inserts', 'ANM sound-field'], colors, corrdict, sigbars=sigbars)
    plt.subplots_adjust(hspace=0.3)
    plt.ylim([-0.015,0.045])
    plt.savefig(savepath / 'fig_AVG3.pdf', bbox_inches='tight', dpi=600)
    plt.savefig(savepath / 'fig_AVG3.png', bbox_inches='tight', dpi=600)


def plot_fig_indivTRFs(resdict, pklatsdict, pkampsdict, savepath, saveflag=False):
    colorsA = [(1,0.6,0.4), (0.8, 0.3, 0.3), (0.4,0.6,1), (0.3, 0.3, 0.8),]
    subjects = list(range(1, 25))
    ylim = [-1.3, 2.2]
    fig = plt.figure(figsize=(20,20))
    plot_individuals(resdict, ['io_click_erp', 'in_click_erp', 's_click_erp'], subjects, colors=[(0.5,0.5,0.5), colorsA[3], colorsA[1]], pklatsdict=pklatsdict, pkampsdict=pkampsdict, corrks= ['in_click_erp', 's_click_erp'], ylim=ylim, fig=fig)
    plt.suptitle('Individual Click ERPs', y=0.92, fontsize=15)
    if saveflag: plt.savefig(savepath / 'fig_indivTRFs_clicks.pdf', bbox_inches='tight', dpi=600)
    if saveflag: plt.savefig(savepath / 'fig_indivTRFs_clicks.png', bbox_inches='tight', dpi=600)

    fig = plt.figure(figsize=(20,20))
    plot_individuals(resdict, ['in_speech_ANnull', 's_speech_ANnull', 'in_speech_AN', 's_speech_AN'], subjects, colors=[(0.5, 0.5, 0.7), (0.7, 0.5, 0.5), colorsA[3], colorsA[1]], pklatsdict=pklatsdict, pkampsdict=pkampsdict, corrks= ['in_speech_AN', 's_speech_AN'], ylim=ylim, fig=fig)
    plt.suptitle('Individual Speech TRFs: ANM Predictor', y=0.92, fontsize=15)
    if saveflag: plt.savefig(savepath / 'fig_indivTRFs_speechAN.pdf', bbox_inches='tight', dpi=600)
    if saveflag: plt.savefig(savepath / 'fig_indivTRFs_speechAN.png', bbox_inches='tight', dpi=600)

    fig = plt.figure(figsize=(20,20))
    plot_individuals(resdict, ['in_speech_RSnull', 's_speech_RSnull', 'in_speech_RS', 's_speech_RS'], subjects, colors=[(0.5, 0.5, 0.7), (0.7, 0.5, 0.5), colorsA[3], colorsA[1]], pklatsdict=pklatsdict, pkampsdict=pkampsdict, corrks=['in_speech_RS', 's_speech_RS'], ylim=ylim, fig=fig)
    plt.suptitle('Individual Speech TRFs: RS Predictor', y=0.92, fontsize=15)
    if saveflag: plt.savefig(savepath / 'fig_indivTRFs_speechRS.pdf', bbox_inches='tight', dpi=600)
    if saveflag: plt.savefig(savepath / 'fig_indivTRFs_speechRS.png', bbox_inches='tight', dpi=600)

    fig = plt.figure(figsize=(20,20))
    corrs = plot_individuals(resdict, ['in_click_erp', 'in_speech_AN'], subjects, colors=[colorsA[3], colorsA[1]], pklatsdict=pklatsdict, pkampsdict=pkampsdict, corrks= ['in_click_erp', 'in_speech_AN'], ylim=ylim, fig=fig)
    if saveflag: plt.savefig(savepath / 'fig_indivTRFs_inserts.pdf', bbox_inches='tight', dpi=600)
    if saveflag: plt.savefig(savepath / 'fig_indivTRFs_inserts.png', bbox_inches='tight', dpi=600)
    plt.figure()
    plt.hist(corrs)

    fig = plt.figure(figsize=(20,20))
    corrs = plot_individuals(resdict, ['s_click_erp', 's_speech_AN'], subjects, colors=[colorsA[3], colorsA[1]], pklatsdict=pklatsdict, pkampsdict=pkampsdict, corrks=['s_click_erp', 's_speech_AN'], ylim=ylim, fig=fig)
    if saveflag: plt.savefig(savepath / 'fig_indivTRFs_speakers.pdf', bbox_inches='tight', dpi=600)
    if saveflag: plt.savefig(savepath / 'fig_indivTRFs_speakers.png', bbox_inches='tight', dpi=600)
    plt.figure()
    plt.hist(corrs)


def plot_avg(resdict, conditions, fig=None, labels=None, title=None, savefile=None, colors=None, y_sigbar=-4*1e-7, ylim=None):
    tcrit = scipy.stats.t

    if labels is None:
        labels = conditions

    if colors is None:
        colors =  [(1,0.6,0.4), (0.8, 0.3, 0.3), (0.4,0.6,1), (0.3, 0.3, 0.8)]

    if fig is None:
        plt.figure()
    for i, k in enumerate(conditions):
        resdict[k] = eel.combine(resdict[k])
        ym = resdict[k].mean('case').x
        ys = resdict[k].std('case').x/np.sqrt(len(resdict[k]))
        clow = ym-ys
        chigh = ym+ys
        print(np.max(ym))
        plt.plot(resdict[k].time.times*1e3, ym, color=colors[i], alpha=0.8, lw=2)
        plt.fill_between(resdict[k].time.times*1e3, clow, chigh, color=colors[i], alpha=0.15)
        
    plt.axhline(0, color='k')
    if ylim is not None:
        plt.ylim(ylim)
    ylabels = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{y:.2f}' for y in ylabels])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))

    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
    plt.legend(custom_lines, [c for c in labels], loc='upper left')
    plt.axvline(0, color='k', linewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uV)')
    plt.grid(b=True, axis='x', which='major', color=(0.7,0.7,0.7))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if title is not None:
        plt.title(title, fontsize=13)
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight', dpi=600)


def plot_corr(ax, ks, labels, colors, corrsA, sigbars=None):
    import matplotlib.patches as mpatches

    patches = []
    for ii, (k, label, color) in enumerate(zip(ks, labels, colors)):
        print(k)
        bplot = ax.boxplot([[corrsA[k][isubj] for isubj in range(len(corrsA[k]))], [corrsA[k+'null'][isubj] for isubj in range(len(corrsA[k]))]], 
                                positions=[ii*2+i*0.5 for i in range(2)], patch_artist=True, widths=0.3)
        bplot['boxes'][0].set_facecolor(color)
        bplot['boxes'][1].set_facecolor([i for i in color[:3]]+[0.1])

        for m in bplot['medians']:
            m.set_color('black')
        patches.append(mpatches.Patch(color=color, label=label))

    ax.set_ylim([-0.015, 0.05])
    ax.set_xticks([i*2+0.25 for i in range(4)])
    ax.set_xticklabels(labels, fontsize=10)

    ax.axhline(0, color='k', linestyle='dashed')
    ax.set_ylabel('Prediction Correlation', fontsize=13)
    ax.set_xlabel('Predictor', fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title('Model Fit', fontsize=14)

    if sigbars is not None:
        for i, sigbar in enumerate(sigbars):
            x1 = sigbar[0]*2+0.1
            x2 = sigbar[1]*2+0.1
            y1 = 0.04-0.018*(len(sigbars)-i-1)/len(sigbars)
            y2 = y1 + 0.001
            xmark = 0.5*(x1+x2)
            ymark = y2 + 0.001
            if sigbar[2] > 0.05:
                color = (0.5,0.5,0.5)
                sigstr = 'n.s.'
                va = 'bottom'
            elif sigbar[2] > 0.01:
                color = (1, 0.5, 0)
                sigstr = '*'
                va = 'center'
            elif sigbar[2] > 0.001:
                color = (1, 0.3, 0)
                sigstr = '**'
                va = 'center'
            else:
                color = (1, 0, 0)
                sigstr = '***'
                va = 'center'
            plt.plot([x1, x2], [y2, y2], color=color)
            plt.plot([x1, x1], [y1, y2], color=color)
            plt.plot([x2, x2], [y1, y2], color=color)
            plt.text(xmark, ymark, sigstr, color=color,  ha='center', va=va)



def plot_indiv_correlations(pklatsdict, pkampsdict, xks, yks, xstrs, ystrs, titles):
    import statsmodels.stats.multitest
    pslats = []
    psamps = []
    tslats = []
    tsamps = []
    for cx, cy, ct in zip(xks, yks, titles):
        xx = pklatsdict[cx]
        yy = pklatsdict[cy]
        t, p = scipy.stats.pearsonr(xx, yy)
        pslats.append(p)
        tslats.append(t)
        
        xx = pkampsdict[cx]
        yy = pkampsdict[cy]
        t, p = scipy.stats.pearsonr(xx, yy)
        psamps.append(p)
        tsamps.append(t)

    corr_xstrs = titles
    [_, pcorrected, _,_] = statsmodels.stats.multitest.multipletests(pslats+psamps)
    pclats = pcorrected[:len(corr_xstrs)]
    pcamps = pcorrected[len(corr_xstrs):]
    print(pclats)
    print(pcamps)
    plt.figure(figsize=(20,8))
    N = len(corr_xstrs)
    colors = [(0.5,0.4,0.8), (0.3,0,0.6)]
    for i, (cx, cy, sx, sy, ct, pclat, pcamp, tlat, tamp) in enumerate(zip(xks, yks, xstrs, ystrs, titles, pclats, pcamps, tslats, tsamps)):
        plt.subplot(2,N,i+1)
        xx = [x*1000 for x in pklatsdict[cx]]
        yy = [y*1000 for y in pklatsdict[cy]]
        plt.scatter(xx, yy, color='k', s=25)
        mn = np.min([np.min(xx), np.min(yy)])*0.95
        mx = np.max([np.max(xx), np.max(yy)])*1.05
        print(mn, mx)
        plt.plot([4.5, 8], [4.5, 8], color='k', linestyle='dashed')
        plt.xlim([4.5, 8])
        plt.ylim([4.5, 8])
        plt.gca().set_yticks([5, 6, 7, 8])
        plt.gca().set_yticklabels([5, 6, 7, 8])
        pstr = 'p<0.001' if pclat < 0.001 else f'p={pclat:.4f}'
        plt.text(6, 8.7, ct, fontsize=16, horizontalalignment='center')
        plt.text(6, 8.2, f'r={tlat:.3f}, {pstr}', fontsize=14, horizontalalignment='center')
        plt.xlabel('Latency [ms]\n'+sx, fontsize=14)
        plt.ylabel(sy+'\nLatency [ms]', fontsize=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)


        plt.subplot(2,N,N+i+1)
        xx = pkampsdict[cx]
        yy = pkampsdict[cy]
        plt.scatter(xx, yy, color='k', s=25)
        mn = np.min([np.min(xx), np.min(yy)])*0.95
        mx = np.max([np.max(xx), np.max(yy)])*1.05
        mn = -0.2
        mx = 2
        pstr = 'p<0.001' if pcamp < 0.001 else f'p={pcamp:.4f}'
        if cx == 'in_speech_AN':
            plt.xlim([-0.2, 2.2])
            plt.ylim([-0.2, 1.5])
            plt.text(1, 1.8, ct, fontsize=16, horizontalalignment='center')
            plt.text(0.9, 1.6, f'r={tamp:.3f}, {pstr}', fontsize=14, horizontalalignment='center')
            plt.gca().set_yticks([0, 0.5, 1, 1.5])
            plt.gca().set_yticklabels([0, 0.5, 1, 1.5])
        elif cy == 'in_speech_AN':
            plt.xlim([-0.2, 1.5])
            plt.ylim([-0.2, 2.2])
            plt.text(0.65, 2.6, ct, fontsize=16, horizontalalignment='center')
            plt.text(0.65, 2.3, f'r={tamp:.3f}, {pstr}', fontsize=14, horizontalalignment='center')
            plt.gca().set_yticks([0, 1, 2])
            plt.gca().set_yticklabels([0, 1, 2])
        else:
            plt.xlim([-0.2, 1.5])
            plt.ylim([-0.2, 1.5])
            plt.text(0.65, 1.8, ct, fontsize=16, horizontalalignment='center')
            plt.text(0.65, 1.6, f'r={tamp:.3f}, {pstr}', fontsize=14, horizontalalignment='center')
            plt.gca().set_yticks([0, 0.5, 1, 1.5])
            plt.gca().set_yticklabels([0, 0.5, 1, 1.5])
        plt.plot([mn,mx], [mn,mx], color='k', linestyle='dashed')

        plt.xlabel('Amplitude [a.u.]\n'+sx, fontsize=14)
        plt.ylabel(sy+'\nAmplitude [a.u.]', fontsize=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    plt.subplots_adjust(hspace=0.7, wspace=0.7)
    print('done')
    statsdict = dict(pslats=pslats, psamps=psamps, tslats=tslats, tsamps=tsamps, pclats=pclats, pcamps=pcamps)
    
    return statsdict



def plot_individuals(resdict, ks, subjects, colors=None, pklatsdict=None, pkampsdict=None, corrks=None, ylim=None, fig=None, nrow=6, ncol=4):
    if colors is None:
        colors = [(1,0.6,0.4), (0.8, 0.3, 0.3), (0.4,0.6,1), (0.3, 0.3, 0.8)]
    corrs = []
    if fig is None:
        plt.figure(figsize=(12,18))
    for i, subject in enumerate(subjects):
        plt.subplot(nrow,ncol,i+1)
        for k, color in zip(ks, colors):
            plt.plot(resdict[k][i].time.times*1000, resdict[k][i].x, color=color, lw=2)
            if pklatsdict is not None and 'null' not in k and 'io' not in k:
                plt.scatter(pklatsdict[k][i]*1000, pkampsdict[k][i], s=80, color='k', marker='x') 
        if corrks is not None:
            corr = np.corrcoef(resdict[corrks[0]][i].sub(time=(0, 0.015)).x, resdict[corrks[1]][i].sub(time=(0, 0.015)).x)[0, 1]
        if i==0:
            custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(ks))]
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if corrks is not None:
            plt.title(f'P{subject:02d}, correlation r={corr:.3f}')
        else:
            plt.title(f'P{subject:02d}')
        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
        plt.axhline(0, color='k')
        plt.axvline(0, color='k')
        plt.grid(b=True, axis='x', which='major', color=(0.5,0.5,0.5), linestyle='--')
        if ylim is not None: plt.ylim(ylim)
        if corrks is not None: corrs.append(corr)
    plt.subplots_adjust(hspace=0.3)

    return corrs