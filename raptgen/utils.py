from typing import Callable, Tuple, List
import GPy
import matplotlib as mpl
from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
import pickle
import torch
from raptgen.models import CNN_PHMM_VAE
from raptgen.data import SingleRound, Result, ProfileHMMSampler, State
from raptgen.visualization import SeqLogoDrawer
from matplotlib.patches import Ellipse
from pathlib import Path
from sklearn.mixture import GaussianMixture
import numpy as np

def plotEllipse(
    ax: plt.Axes,
    mu: np.ndarray,
    coval: np.ndarray,
    enrich: float | None = None,
    ) -> plt.Axes:
    """二次元の平均ベクトルと分散共分散行列を引数に取り， `ax` に該当のガウス分布を示す楕円を書き込む"""
    xcenter, ycenter = mu

    vals, vecs = np.linalg.eigh(coval)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    nstd = 2
    w, h = 2 * nstd * np.sqrt(vals)

    ell = Ellipse(
        xy        = (xcenter, ycenter),
        width     = w,
        height    = h,
        angle     = theta,
        linewidth = 2,
        fill      = False,
        zorder    = 3,
        edgecolor = plt.get_cmap('cool')(enrich) if enrich != None else "orange",
        facecolor = "none"
    )

    ax.add_patch(ell)

    return ax

def plotReads(
    axes:  plt.Axes,
    label: str,
    data:  np.ndarray,
    color: str
    ) -> PathCollection:
    """`data` に示される座標を `ax` にプロットする"""
    assert data.ndim == 2
    assert data.shape[1] == 2
    axes.set_title(label, fontsize=24)
    axes.set_xlim(-3.5, 3.5)
    axes.set_ylim(-3.5, 3.5)
    sc = axes.scatter(*data.T, s=1, color=color)
    return sc

def plotGMM(figure: Figure,
            GMM_model: GaussianMixture
            ) -> Figure:
    axes = figure.axes
    GMM_means = GMM_model.means_
    GMM_covals = GMM_model.covariances_
    for ax in axes:
        for mean, coval in zip(GMM_means, GMM_covals):
            plotEllipse(ax = ax, 
                        mu = mean,
                        coval = coval, 
                        enrich = None)
    return figure

def makeWeblogoFigure(result: Result,
                      num_q: int,
                      **kwargs
                      ) -> Tuple[Figure, plt.Axes]:
    assert num_q > 1
    figure, axes = plt.subplots(nrows=num_q, ncols=num_q, **kwargs)
    x = np.linspace(-3.5, 3.5, num_q)
    y = np.linspace(3.5, -3.5, num_q)
    points_x, points_y = np.meshgrid(x, y)
    points = np.vstack((points_x.flatten(), points_y.flatten())).T

    for index, ax in enumerate(axes.flatten()):
        a, e_m = result.model.decoder(torch.Tensor([points[index]]))
        a = a.detach().numpy()[0]
        e_m = e_m.detach().numpy()[0]
        proba = list()
        for i, state in ProfileHMMSampler(a, e_m, proba_is_log=True).most_probable()[0]:
            if 0 < i <= e_m.shape[0]:
                if state == State.M:
                    proba.append(np.exp(e_m[i-1]))
                elif state == State.I:
                    proba.append(np.ones((4))*0.25)
        SeqLogoDrawer().draw_logo(np.stack(proba).T, ax=ax)
        ax.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
    
    figure.tight_layout()

    for ax, col in zip(axes[0], x):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], y):
        ax.set_ylabel(row, rotation=0, size='large')
    
    return figure, axes