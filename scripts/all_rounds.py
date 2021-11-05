## run 10 motif split simulation script
import logging
from re import S

import click 
import numpy as np
from pathlib import Path

import torch

from raptgen.models import CNN_PHMM_VAE
from raptgen.data import SingleRound, Result, read_fastq
from sklearn.mixture import GaussianMixture

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = str(Path(f"{dir_path}/../out/gmm").resolve())

@click.command(help='Obtains mean latent vectors of sequences for single and all rounds, applies Gaussian mixture models of specified number of classes to all rounds latent data, and for each of single round data, calculates the probabilties of belonging to each class. these mean vectors and probabilities are stored as csv under --save-dir directory.',
                    context_settings=dict(show_default=True))
@click.argument("all-round-path",
                    type=click.Path(exists = True))
@click.argument("model-path",
                    type=click.Path(exists = True))
@click.option("--seq-path", "-s",
                    type=click.Path(exists = True),
                    multiple=True)
@click.option("--use-cuda/--no-cuda",
                  help    = "use cuda if available",
                  is_flag = True,
                  default = True)
@click.option("--cuda-id",
                  help    = "the device id of cuda to run",
                  type    = int, default = 0)
@click.option("--gmm-save-dir",
                  help    = "path to save the gmm model and probs of seqs in each round",
                  type    = click.Path(),
                  default = default_path)
@click.option("--latent-save-dir",
                  help    = "path to save latent mean vectors of seqs in each round",
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
@click.option("--num-components",
                  help    = "the number of class the Gaussian mixture model has",
                  type    = int,
                  default = 10)
@click.option("--calc-times",
                  help    = "the number of times calculation of GMM does (not num of iter EM)",
                  type    = int,
                  default = 100)
@click.option("--min-count",
                  help    = "minimal counts considered on all round GMM",
                  type    = int,
                  default = 1)
def main(all_round_path, model_path, seq_path, cuda_id, use_cuda, gmm_save_dir, latent_save_dir, fwd, rev, num_components, calc_times, min_count):
    # get "all_rounds.py"-specific logger
    logger = logging.getLogger(__name__)
    
    # make a directory to save csv and gmm-model available
    logger.info(f"opening saving directory: {gmm_save_dir}")
    gmm_save_dir = Path(gmm_save_dir).expanduser().resolve()
    gmm_save_dir.mkdir(exist_ok = True, parents=True)
    logger.info(f"opening saving directory: {latent_save_dir}")
    latent_save_dir = Path(latent_save_dir).expanduser().resolve()
    latent_save_dir.mkdir(exist_ok = True, parents=True)

    # initialization of SingleRound class does not run the model, just loads sequences from path or as raw argments.
    experiment = SingleRound(
        path            = str(Path(all_round_path).expanduser().resolve()),
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
    all_round_result = Result(
        model,
        experiment           = experiment,
        path_to_save_results = gmm_save_dir,
        load_if_exists       = True,
        min_count            = min_count
    )

    # calc every mu vectors of probablistic function in latent space from the sequence data
    logger.info(f"calculating mean vectors of latent pdf defined by each SELEX reads")
    all_round_result.get_mean_vectors_from_experiment(get_raw_seq=False)
    if not Path(f"{latent_save_dir}/all_rounds_latent.csv").exists():
        np.savetxt(f"{latent_save_dir}/all_rounds_latent.csv", all_round_result.mus, delimiter=',')
    
    all_round_result.calc_gmm(dim = num_components, calc_times=calc_times)
    logger.info(f"loading calculated GMM")
    gmm_model: GaussianMixture = all_round_result.gmm

    
    ## GMM MODEL, CNN-pHMM-VAE MODEL LOADED
    

    for single_round_path in seq_path:
        if not ( Path(f"{latent_save_dir}/{Path(single_round_path).stem}.csv").exists() 
                and Path(f"{gmm_save_dir}/{Path(single_round_path).stem}_probs.csv").exists() ):

            # each probability of classes are predicted here
            single_round_result = Result(
                model,
                experiment           = SingleRound(
                    path            = single_round_path,
                    forward_adapter = all_round_result.experiment.forward_adapter,
                    reverse_adapter = all_round_result.experiment.reverse_adapter
                ),
                path_to_save_results = gmm_save_dir,
                load_if_exists       = True,
                min_count            = 1
            )

            single_round_mus, single_round_seqs = single_round_result.get_mean_vectors_from_experiment(get_raw_seq=True)
            _, all_round_seqs = all_round_result.get_mean_vectors_from_experiment(get_raw_seq=True)

            boolean_map = [seq in all_round_seqs for seq in single_round_seqs]
            latent_seqs_mus_single_round = single_round_mus[boolean_map]

            logger.info(f"predicting each probabilities of {num_components} classes for single round reads")
            
            X_probs: np.ndarray = gmm_model.predict_proba(latent_seqs_mus_single_round)
            np.savetxt(f"{latent_save_dir}/{Path(single_round_path).stem}.csv", latent_seqs_mus_single_round, delimiter=',')
            np.savetxt(f"{gmm_save_dir}/{Path(single_round_path).stem}_probs.csv", X_probs, delimiter=',')
    
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
