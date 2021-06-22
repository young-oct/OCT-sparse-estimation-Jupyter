# -*- coding: utf-8 -*-
# @Time    : 2021-02-02 10:09 p.m.
# @Author  : young wang
# @FileName: OCT_dictCompare.py
# @Software: PyCharm


from pathlib import Path
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sporco import prox
from sporco.admm import cbpdn
from misc.postprocessing import intensity_norm
from skimage.exposure import match_histograms
import copy
from misc import quality
from matplotlib.patches import Rectangle
import matplotlib
from skimage.metrics import structural_similarity as ssim
from sporco import metric

def get_background(x, y, width, height):
    space = [Rectangle((x, y), width, height, linewidth=2, edgecolor='cyan', fill=False)]
    return space


def get_homogeneous(x, y, width, height):
    space = [Rectangle((x, y), width, height, linewidth=2, edgecolor='red', fill=False)]
    return space


def get_artifact(x, y, width, height):
    space = [Rectangle((x, y), width, height, linewidth=2, edgecolor='green', fill=False)]
    return space

np.seterr(divide='ignore', invalid='ignore')
matplotlib.rcParams.update(
    {
        'font.size': 18,
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)


path = Path('/Users/youngwang/Desktop/Data/paper/PSF')
S_PATH = '/Users/youngwang/Desktop/Data/paper/Data/ear'

files = []

for i in path.iterdir():
    files.append(i)
#rearrange dic order
files.sort(reverse=True)
files.pop(0)

dic = np.zeros((330, 1, len(files)), complex)
#
for i in range(len(files)):
    with open(files[i], 'rb') as f:
        dic[:, :, i] = pickle.load(f)
        f.close()
#
with open(S_PATH, 'rb') as f:
    s = pickle.load(f).T
    f.close()

# pre-processing data
for i in range(s.shape[1]):
    # (1) remove the DC term of each A-line by
    # subtracting the mean of the A-line
    s[:, i] -= np.mean(s[:, i])

# (2) remove background noise: minus the frame mean
s -= np.mean(s, axis=1)[:, np.newaxis]
s_removal = copy.deepcopy(s)

# (3) l2 norm data and save the scaling factor
l2f = prox.norm_l2(s_removal,axis=0).squeeze()
for i in range(s_removal.shape[1]):
    s_removal[:,i] /= l2f[i]

s_removal_log = 20 * np.log10(abs(s.T))
s_log_norm = intensity_norm(s_removal_log).T

Maxiter = 20
opt_par = cbpdn.ConvBPDN.Options({'FastSolve': False, 'Verbose': False, 'StatusHeader': False,
                                  'MaxMainIter': Maxiter, 'RelStopTol': 5e-5, 'AuxVarObj': True,
                                  'RelaxParam': 1.515, 'AutoRho': {'Enabled': True}})
index = 6500
test = s_removal[:, index]
lmbda = 3.6e-2

eps = 1e-14

# determine the scaling parameters
scale = np.zeros((len(files)))
sparse = np.zeros((330, 10240, len(files)))
x_line = np.zeros((330, len(files)))
sparisty = np.zeros(len(files))

x_L = np.zeros((330, 10240, len(files)))
for i in range(len(files)):
    D0 = dic[:,:,i]
    b = cbpdn.ConvBPDN(D0, test, lmbda, opt=opt_par, dimK=None, dimN=1)
    x = b.solve()
    x = x.squeeze()
    x = np.roll(x, np.argmax(D0), axis=0)
    scale[i] = np.max(abs(x)) / np.max(abs(test))
    # x_line[:,i] = abs(x)/scale[i]

    b = cbpdn.ConvBPDN(D0, s_removal, lmbda, opt=opt_par, dimK=1, dimN=1)
    x = b.solve()
    x = x.squeeze()
    x = np.roll(x, np.argmax(D0), axis=0)

    sparisty[i] = np.count_nonzero(x) * 100 / x.size
#
    for j in range(x.shape[1]):
        x[:, j] *= l2f[j]

    x_log = x.T
    # rescale the sparse solution
    for j in range(s.shape[1]):
        x_log[j, :] = abs(x_log[j, :]) / scale[i]

    x_line[:, i] = abs(x_log.T[:, index])
    x_L[:,:,i] =  abs(x_log.T)

    x_log = 20 * np.log10(abs(x_log))
    # # display rescaling, forcing -inf to be a 20*np.log10(esp)
    x_log_corr = np.where(x_log < 20 * np.log10(eps), 20 * np.log10(eps), x_log)

    temp = intensity_norm(x_log_corr).T

    match = match_histograms(temp, s_log_norm, multichannel=False)
    sparse[:,:,i] = np.where(match <= match.min(), 0, match)

vmax = 255
vmin = 120
width, height = (2000, 95)
background = [[3500, 0, width, height-30]]
homogeneous = [[2500, 125, width, height]]
aspect = sparse.shape[1]/sparse.shape[0]

fig, ax = plt.subplots(3, len(files)+1,figsize=(16, 9),constrained_layout=True)
fig.suptitle('%d dB - %d dB' % (vmax, vmin))
ax[0, 0].imshow(s_log_norm, 'gray', aspect=s_log_norm.shape[1] / s_log_norm.shape[0], vmax=vmax, vmin=vmin)
ax[0, 0].set_axis_off()
ax[0, 0].set_title('original')
ax[0, 0].axvline(x=index, linewidth=2, color='orange')
ax[0, 0].set_xlabel('axial depth(pixels)')
for k in range(len(background)):
    for j in get_background(*background[k]):
        ax[0, 0].add_patch(j)
for k in range(len(homogeneous)):
    for j in get_homogeneous(*homogeneous[k]):
        ax[0, 0].add_patch(j)

ho_original = quality.ROI(*homogeneous[0], s_log_norm)
ba_original = quality.ROI(*background[0], s_log_norm)
roi_original = np.where(ho_original<ba_original.mean(),0,ho_original)
ref = np.count_nonzero(roi_original) / roi_original.size

ax[1, 0].imshow(roi_original,'gray',aspect =roi_original.shape[1]/roi_original.shape[0],vmax = vmax, vmin = vmin)
ax[1, 0].set_axis_off()
ax[1, 0].annotate('', xy=(1200, 10), xycoords='data',
            xytext=(900, 5), textcoords='data',
            arrowprops=dict(facecolor='red', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
ax[2, 0].plot(abs(s[:, index]))

roi_sparse = np.zeros((height,width,len(files)))
roi_per = np.zeros(len(files))
loss = np.zeros(len(files))
ssim_index = np.zeros(len(files))
gmsd = np.zeros(len(files))
for i in range(len(files)):

    temp = sparse[:,:,i]
    aspect = sparse[:,:,i].shape[1]/sparse[:,:,i].shape[0]
    ax[0, i + 1].imshow(temp, 'gray', aspect=aspect, vmax=vmax, vmin=vmin)
    ax[0, i + 1].axvline(x=index, linewidth=2, color='orange')
    ax[0, i + 1].set_axis_off()
    if i == 0:
        ax[0, i + 1].set_title('full frame')
    else:
        ax[0, i + 1].set_title('20% frame')

    for k in range(len(background)):
        for j in get_background(*background[k]):
            ax[0, i + 1].add_patch(j)

    for k in range(len(homogeneous)):
        for j in get_homogeneous(*homogeneous[k]):
            ax[0, i + 1].add_patch(j)

    ho = quality.ROI(*homogeneous[0], temp)
    ba = quality.ROI(*background[0], temp)
    roi_sparse[:, :, i] = np.where(ho < ba.mean(), 0, ho)

    roi_per[i] = np.count_nonzero(roi_sparse[:, :, i]) / roi_sparse[:, :, i].size

    loss[i] = (ref - roi_per[i])*100

    ssim_index[i] = ssim(roi_sparse[:, :, i],roi_original, data_range=255.0)
    gmsd[i] = metric.gmsd(roi_original,roi_sparse[:, :, i])

    aspect = width / height

    ax[1, i + 1].imshow(roi_sparse[:, :, i],'gray',aspect =aspect,vmax = vmax, vmin = vmin)
    ax[1, i + 1].annotate('', xy=(1200, 10), xycoords='data',
            xytext=(900, 5), textcoords='data',
            arrowprops=dict(facecolor='red', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
    ax[1, i + 1].set_axis_off()

    textstr = '\n'.join((
        r'gmsd  ''\n'
        r'%.2f' %  (gmsd[i],),
        r'ssim  ''\n'
        r'%.2f' % (ssim_index[i],),
        r'relative loss ''\n'
        r'%.2f%%' % (loss[i],),
        r'sparisty' '\n'
        r'%.2f%%' % (sparisty[i],)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    ax[2, i + 1].text(0.05, 0.98, textstr, transform=ax[2, i + 1].transAxes, fontsize=12,
                      verticalalignment='top', bbox=props)
    textstr = '\n'.join((
        r'gmsd  ''\n'
        r'%.2f' %  (gmsd[i],),
        r'ssim  ''\n'
        r'%.2f' % (ssim_index[i],),
        r'relative loss ''\n'
        r'%.2f%%' % (loss[i],),
        r'sparisty' '\n'
        r'%.2f%%' % (sparisty[i],)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    ax[2, i + 1].text(0.05, 0.98, textstr, transform=ax[2, i + 1].transAxes, fontsize=12,
                      verticalalignment='top', bbox=props)
    ax[2, i + 1].plot(x_line[:,i])

plt.show()
