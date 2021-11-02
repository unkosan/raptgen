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

def normalization(array: np.ndarray):
    max_value = array.max()
    min_value = array.min()
    if max_value - min_value == 0:
        raise Exception("normalization: input data is uniform. can't normalize this.")
    return (array - min_value) / (max_value - min_value)

def plotEllipse(ax:     plt.Axes,
                mu:     np.ndarray,
                coval:  np.ndarray,
                enrich: float or None = None) -> plt.Axes:
    xcenter, ycenter = mu

    vals, vecs = np.linalg.eigh(coval)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    nstd = 2
    w, h = 2 * nstd * np.sqrt(vals)

    ell = Ellipse(xy        = (xcenter, ycenter),
                  width     = w,
                  height    = h,
                  angle     = theta,
                  linewidth = 2,
                  fill      = False,
                  zorder    = 3,
                  edgecolor = plt.get_cmap('cool')(enrich) if enrich != None else "orange",
                  facecolor = "none")

    ax.add_patch(ell)

    return ax

def loadSelexData(single_round_csv_name_gen: Callable,
                  range: List[int],
                  min_counts: int or None = None) -> dict:
    data = dict()
    for i in range:
        data[f"{i} round"] = np.loadtxt(fname = str(single_round_csv_name_gen(i)),
                                        delimiter=",")
    
    if min_counts != None:
        all_round = np.concatenate(list(data.values()), axis=0)

        all_round, counts = np.unique(all_round, axis=0, return_counts=True)
        filtered_points = all_round[counts >= min_counts]
        for i in range:
            boolean_map = tuple([(data[f"{i} round"][j] == filtered_points).all(axis=1).any()
                        for j in np.arange(data[f"{i} round"].shape[0])])
            boolean_map = np.array(boolean_map, dtype=bool)
            data[f"{i} round"] = data[f"{i} round"][boolean_map]
        
    return data

def loadProbSums(single_round_probs_csv_name_gen: Callable,
                 range: list) -> dict:
    probs_sum = dict()
    for i in range:
        probs_data = np.loadtxt(fname=str(single_round_probs_csv_name_gen(i)),
                                delimiter=",", dtype=float)
        probs_data = probs_data.sum(axis=0)
        probs_sum[f"{i} round"] = probs_data
    
    return probs_sum

def loadResult(all_fastq_path: Path,
               vae_model_path: Path,
               device: str = "cuda:3") -> Result:
    """info: This function will take few minutes"""
    experiment = SingleRound(
        path            = str(all_fastq_path),
        forward_adapter = None,
        reverse_adapter = None
    )
    target_len = experiment.random_region_length
    model = CNN_PHMM_VAE(target_len, embed_size=2)
    model.load_state_dict(torch.load(
        str(vae_model_path),
        map_location = device
    ))
    result = Result(
        model,
        experiment           = experiment,
        path_to_save_results = None,
        load_if_exists       = True
    )
    return result

def loadMeasuredData(result: Result,
                     bind_csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    coords = list()
    scores = list()
    with bind_csv_path.open("r") as f:
        lines = f.read().splitlines()
    for line in lines:
        sequence, score = line.split(",")
        coords.append(result.embed_sequences(sequence).to('cpu').detach().numpy().copy()[0])
        scores.append(score)
    coords = np.array(coords, dtype=float)
    scores = np.array(scores, dtype=float)
    return coords, scores

def plotReads(axes:  plt.Axes,
              label: str,
              data:  np.ndarray,
              color: str) -> PathCollection:
    assert data.ndim == 2
    assert data.shape[1] == 2
    axes.set_title(label, fontsize=24)
    axes.set_xlim(-3.5, 3.5)
    axes.set_ylim(-3.5, 3.5)
    sc = axes.scatter(*data.T, s=1, color=color)
    return sc

def plotReadsAllRound(selex_data: dict,
                      figure: Figure,
                      color: str) -> Figure:
    axes = figure.axes
    assert len(axes) == len(selex_data.keys())
    for ax, (key, value) in zip(axes, selex_data.items()):
        plotReads(axes = ax, label = key, data = value, color = color)
    return figure

def loadGMM(gmm_model_path: Path) -> GaussianMixture:
    with gmm_model_path.open("rb") as f:
        gmm = pickle.load(f)
    return gmm

def makeGP(coords: np.ndarray,
           scores: np.ndarray,
           kernel: GPy.kern.src.kern.Kern
           ) -> GPy.core.gp.GP:
    model = GPy.models.GPRegression(coords,
                                    np.reshape(scores, newshape=(-1, 1))
    )
    return model

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

def plotGMMwithMeasuredVals(figure: Figure,
                            measured_coords: np.ndarray,
                            measured_scores: np.ndarray,
                            GMM_model: GaussianMixture,
                            GMM_enriches: np.ndarray
                            ) -> Figure:
    measured_scores = measured_scores.flatten()
    GMM_enriches = GMM_enriches.flatten()
    GMM_enrich_vmax = GMM_enriches.max()
    GMM_enrich_vmin = GMM_enriches.min()
    GMM_enriches = normalization(GMM_enriches)
    GMM_means = GMM_model.means_
    GMM_covals = GMM_model.covariances_

    assert measured_coords.ndim == 2
    assert measured_coords.shape[1] == 2
    assert measured_coords.shape[0] == measured_scores.shape[0]
    assert GMM_enriches.shape[0] == GMM_means.shape[0] == GMM_covals.shape[0]

    axes = figure.axes
    sc = None
    for ax in axes:
        for mean, coval, enrich in zip(GMM_means, GMM_covals, GMM_enriches):
            plotEllipse(ax = ax,
                        mu = mean, 
                        coval=coval,
                        enrich=enrich)

        sc = ax.scatter(*measured_coords.T,
                        s    = 15,
                        cmap = 'viridis',
                        c    = measured_scores)
    
    figure.subplots_adjust(right=0.75)
    figsize = figure.get_size_inches()
    figure.set_size_inches(
        figsize[0] * 1.25,
        figsize[1]
    )
        
    cbar_ax_GMM = figure.add_axes([0.79, 0.15, 0.02, 0.7])
    cbar_ax_GMM.tick_params(labelsize=20)
    figure.colorbar(
        cm.ScalarMappable(
            norm = mpl.colors.Normalize(vmax = GMM_enrich_vmax,
                                        vmin = GMM_enrich_vmin),
            cmap = plt.get_cmap('cool')
        ),
        cax = cbar_ax_GMM
    )

    cbar_ax_sc = figure.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar_ax_sc.tick_params(labelsize=20)
    figure.colorbar(sc, cax = cbar_ax_sc)

    return figure

def plotGPwithMeasuredVals(figure: Figure,
                           measured_coords: np.ndarray,
                           measured_scores: np.ndarray,
                           GP_model: GPy.core.gp.GP
                           ) -> Figure:
    measured_scores = measured_scores.flatten()

    x = np.linspace(-3.5, 3.5, 100)
    y = np.linspace(-3.5, 3.5, 100)
    mesh_x, mesh_y = np.meshgrid(x, y)

    pred_vals = GP_model.predict(np.vstack((mesh_x.flatten(), mesh_y.flatten())).T)

    axes = figure.axes
    ct = None
    sc = None
    for ax in axes:
        ct = ax.contour(mesh_x, mesh_y, pred_vals[0].reshape(mesh_x.shape), cmap='cool')
        sc = ax.scatter(*measured_coords.T, s=15, cmap='viridis', c=measured_scores)
    
    figure.subplots_adjust(right=0.75)
    figsize = figure.get_size_inches()
    figure.set_size_inches(
        figsize[0] * 1.25,
        figsize[1]
    )
        
    cbar_ax_ct = figure.add_axes([0.79, 0.15, 0.02, 0.7])
    cbar_ax_ct.tick_params(labelsize=20)
    figure.colorbar(ct, cax = cbar_ax_ct)

    cbar_ax_sc = figure.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar_ax_sc.tick_params(labelsize=20)
    figure.colorbar(sc, cax = cbar_ax_sc)

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