from raptgen.core.algorithms import *
from raptgen.core.preprocessing import *
from raptgen.core.train import *
from torch import optim
import pickle
import click

@click.command(
    help = 'run experiment with real data', 
    context_settings = dict(show_default = True))
@click.option("--data-path",
    help = 'accepts only fasta or fastq',
    type = click.Path(exists = True),
    required = True)
@click.option("--model-save-path",
    type = click.Path(),
    required = True)
@click.option("--score-save-path",
    type = click.Path(),
    required = True)
@click.option("--seed",
    type = int,
    default = np.random.randint(0, 3141592653589793))
@click.option("--device-name",
    type = str, 
    default = "cpu")
@click.option("--fwd-adapter",
    type = str,
    required = True)
@click.option("--rev-adapter",
    type = str,
    required = True)
@click.option("--target-length",
    type = int,
    required = True)
@click.option("--tolerance",
    type = int,
    required = True)
@click.option("--min-count",
    type = int,
    required = True)
@click.option("--embed-size",
    type = int,
    required = True)
@click.option("--epochs",
    type = int,
    required = True)
@click.option("--force-matching-epochs",
    type = int,
    default = None)
@click.option("--beta-schedule-epochs",
    type = int,
    default = None)
@click.option("--early-stop-threshold",
    type = int,
    default = 50)
def main(
    data_path: str,
    model_save_path: str,
    score_save_path: str,
    seed: int,
    device_name: str,
    fwd_adapter: str,
    rev_adapter: str,
    target_length: int,
    tolerance: int,
    min_count: int,
    embed_size: int,
    epochs: int,
    force_matching_epochs: int,
    beta_schedule_epochs: int,
    early_stop_threshold: int,
) -> None:

    set_seed(seed)

    r_len = target_length - len(fwd_adapter) - len(rev_adapter)

    df = read_SELEX_data(
        filepath = data_path,
        filetype = Path(data_path).suffix[1:],
        is_biopython_format = False,
    )

    df = df[df['Sequence'].apply(
        lambda seq: default_filterfunc(
            seq,
            fwd_adapter = fwd_adapter,
            rev_adapter = rev_adapter,
            target_length = target_length,
            tolerance = tolerance,
        )
    )] # filter with t_len ± tolerance

    se_unique = df['Sequence'].value_counts()
    se_unique = se_unique[se_unique >= min_count]
    # filter with minimum count

    se_encode = pd.Series(se_unique.index).apply(
        lambda seq: default_cutfunc(
            read = seq, 
            fwd_adapter = fwd_adapter, 
            rev_adapter = rev_adapter,
        )
    ).apply( # cut adapters
        lambda seq: ID_encode(
            seq,
            right_padding = r_len + tolerance - len(seq)
        )
    ) # encode data

    train_loader, test_loader = get_dataloader(
        ndarray_data = np.array(se_encode.to_list()),
        test_size = 0.1,
        batch_size = 512,
        train_test_shuffle = True,
        use_cuda = (device_name != "cpu"),
        num_workers = 2,
        pin_memory = False,
    )

    device = torch.device(device_name)

    model = CNN_PHMM_VAE(
        motif_len = r_len,
        embed_size = embed_size,
    ).to(device)

    optimizer = optim.Adam(model.parameters())

    model_trained, df_trained = train_VAE(
        num_epochs = epochs,
        model = model,
        train_loader = train_loader,
        test_loader = test_loader,
        optimizer = optimizer,
        device = device,
        early_stop_threshold = early_stop_threshold,
        beta_schedule = (beta_schedule_epochs != None),
        beta_threshold = beta_schedule_epochs,
        force_matching = (force_matching_epochs != None),
        force_epochs = force_matching_epochs,
        show_tqdm = False,
    )

    with Path(model_save_path).open("wb") as handle:
        pickle.dump(obj = model_trained.state_dict(), file = handle)

    df_trained.to_pickle(path = score_save_path)

    torch.cuda.empty_cache()

    print(f"""Training finished with the following settings
    SELEX data path: {data_path}
    Model saved path: {model_save_path}
    Score saved path: {score_save_path}
    Seed value: {seed}
    Device name: {device_name}
    Forward adapter: {fwd_adapter}
    Reverse adapter: {rev_adapter}
    Target length: {target_length}
    Tolerance value: {tolerance}
    Minimum count: {min_count}
    Embedding size: {embed_size} dimension
    Maximum epochs: {epochs} epochs
    Force matching epochs: {str(force_matching_epochs) + " epochs" if force_matching_epochs != None else None}
    Beta scheduling epochs: {str(beta_schedule_epochs) + " epochs" if beta_schedule_epochs != None else None}
    Early-stop Threshold: {early_stop_threshold} epochs""")

if __name__ == "__main__":
    main()