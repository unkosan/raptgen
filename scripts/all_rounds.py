## run 10 motif split simulation script
import logging
from re import S

import click 
import numpy as np
from pathlib import Path

import torch

from raptgen.models import CNN_PHMM_VAE
from raptgen.data import SingleRound, Result
from sklearn.mixture import GaussianMixture

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = str(Path(f"{dir_path}/../out/gmm").resolve())

@click.command(help='Obtains mean latent vectors of sequences for single and all rounds, applies Gaussian mixture models of specified number of classes to all rounds latent data, and for each of single round data, calculates the probabilties of belonging to each class. these mean vectors and probabilities are stored as csv under --save-dir directory.',
                    context_settings=dict(show_default=True))
@click.argument("all-rounds-seq-path",
                    type=click.Path(exists = True))
@click.argument("round-seq-path",
                    type=click.Path(exists = True))
@click.argument("model-path",
                    type=click.Path(exists = True))
@click.option("--use-cuda/--no-cuda",
                  help    = "use cuda if available",
                  is_flag = True,
                  default = True)
@click.option("--cuda-id",
                  help    = "the device id of cuda to run",
                  type    = int, default = 0)
@click.option("--save-dir",
                  help    = "path to save results",
                  type    = click.Path(),
                  default = default_path)
@click.option("--fwd",
                  help    = "forward adapter",
                  type    = str,
                  default = None)
@click.option("--rev",
                  help    = "reverse adapter",
                  type    = str,
                  default = None)
@click.option("--num-class",
                  help    = "the number of class the Gaussian mixture model has",
                  type    = int,
                  default = 10)
def main(all_round_path, seq_path, model_path, cuda_id, use_cuda, save_dir, fwd, rev, num_components):
    # get "all_rounds.py"-specific logger
    logger = logging.getLogger(__name__)
    
    # make a directory to save csv and params of pytorch-model available
    logger.info(f"opening saving directory: {save_dir}")
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(exist_ok = True, parents=True)

    # initialization of SingleRound class does not run the model, just loads sequences from path or as raw argments.
    experiment = SingleRound(
        path            = all_round_path,
        forward_adapter = fwd,
        reverse_adapter = rev
    )
    # "target_len" is the length SELEX read minus that of fwd and rev adapter
    target_len = experiment.random_region_length
    # restore pytorch model with tuned parameters from "modelpath"
    logger.info(f"loading CNN-pHMM-VAE model...")
    model = CNN_PHMM_VAE(target_len, embed_size=2)
    device = torch.device(f"cuda:{cuda_id}" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # get result instance of "all_rounds"
    result = Result(
        model,
        experiment           = experiment,
        path_to_save_results = save_dir,
        load_if_exists       = True
    )

    # calc every mu vectors of probablistic function in latent space from the sequence data
    logger.info(f"calculating mean vectors of latent pdf defined by each SELEX reads")
    result.get_mean_vectors_from_experiment(get_raw_seq=False)
    if not Path(f"{save_dir}/all_rounds_latent.csv").exists():
        np.savetxt(f"{save_dir}/all_rounds_latent.csv", result.mus, delimiter=',')
    
    # restore gmm model from "gmm.pkl", or calculate each var and mean of Gaussian mixuture.
    # caution: if a gmm.pkl exists in "save_dir", discards the value of "num_class" and gets the gmm model from the gmm.pkl
    result.calc_gmm(dim = num_components)
    logger.info(f"loading calculated GMM")
    gmm_model: GaussianMixture = result.gmm

    
    ## GMM MODEL, CNN-pHMM-VAE MODEL LOADED
    

    if not Path(f"{save_dir}/{Path(seq_path).stem}").exists():
    # each probability of classes are predicted here
        result = Result(
            model,
            experiment           = SingleRound( path            = seq_path,
                                                forward_adapter = fwd,
                                                reverse_adapter = rev ),
            path_to_save_results = save_dir,
            load_if_exists       = True
        )
        logger.info(f"predicting each probabilities of {num_components} classes for single round reads")
        latent_seqs_mus_single_round = result.get_mean_vectors_from_experiment()
        X_probs: np.ndarray = gmm_model.predict_proba(latent_seqs_mus_single_round)
        np.savetxt(f"{save_dir}/{Path(seq_path).stem}.csv", latent_seqs_mus_single_round, delimiter=',')
        np.savetxt(f"{save_dir}/{Path(seq_path).stem}_probs.csv", X_probs, delimiter=',')
    
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
