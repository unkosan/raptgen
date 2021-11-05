## run 10 motif split simulation script
import logging
from re import S

import click 
import numpy as np
from pathlib import Path

import pickle
import torch
from tqdm.std import tqdm

from raptgen.models import CNN_PHMM_VAE
from raptgen.data import SingleRound, Result
from sklearn.mixture import BayesianGaussianMixture

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = str(Path(f"{dir_path}/../out/gmm").resolve())

@click.command(help='Obtains mean latent vectors of sequences for single and all rounds, applies Gaussian mixture models of specified number of classes to all rounds latent data, and for each of single round data, calculates the probabilties of belonging to each class. these mean vectors and probabilities are stored as csv under --save-dir directory.',
                    context_settings=dict(show_default=True))
@click.argument("points-data-path",
                    type=click.Path(exists = True))
@click.option("--gmm-save-dir",
                  help    = "path to save the gmm model and probs of seqs in each round",
                  type    = click.Path(),
                  default = default_path)
@click.option("--calc-times",
                  help    = "the number of times calculation of GMM does (not num of iter EM)",
                  type    = int,
                  default = 100)
@click.option("--num-components",
                  help = "maximum num of components",
                  type = int,
                  default = 150)
def main(points_data_path, gmm_save_dir, calc_times, num_components):
    # get "all_rounds.py"-specific logger
    logger = logging.getLogger(__name__)
    
    # make a directory to save csv and gmm-model available
    logger.info(f"opening saving directory: {gmm_save_dir}")
    gmm_save_dir = Path(gmm_save_dir).expanduser().resolve()
    gmm_save_dir.mkdir(exist_ok = True, parents=True)

    logger.info(f"opening data directory: {points_data_path}")

    data = np.loadtxt(fname = points_data_path, delimiter = ",")
    
    best_score = -np.inf
    pbar = tqdm(range(calc_times))
    for i in pbar:
        bgmm = BayesianGaussianMixture(
            n_components = num_components,
            covariance_type = "full"
        ).fit(data)
        if bgmm.score(data) > best_score:
            best_score = bgmm.score(data)
            best_bgmm = bgmm
        pbar.set_description(
            "[" + "⠸⠴⠦⠇⠋⠙"[i % 6] + "]" + f"{best_score:.2f}")

    gmm_save_dir = Path(f"{gmm_save_dir}/bgmm.pkl")
    with gmm_save_dir.open("wb") as f:
        pickle.dump(best_bgmm, f)
    
    logger.info(f"finished")


if __name__ == "__main__":
    Path("./.log").mkdir(parents=True, exist_ok=True)
    formatter = '%(levelname)s : %(name)s : %(asctime)s : %(message)s'
    logging.basicConfig(
        filename='.log/logger.log',
        level=logging.DEBUG,
        format=formatter)
        
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    main()
