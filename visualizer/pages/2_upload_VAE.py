import io
from typing import Any, Dict, List, Sequence
import dash
from raptgen.core.algorithms import *
from raptgen.core.preprocessing import *
from raptgen.core.train import *
import pandas as pd
import pickle
from raptgen.core.algorithms import CNN_PHMM_VAE
import dash_bootstrap_components as dbc
import tempfile
import base64
import re

from dash import html, dcc, Input, Output, State, callback

import plotly.graph_objects as go
import datetime
from datetime import date
import numpy as np

from . import util
from . import plot
from .message_broker import MessageBroker

dash.register_page(
    __name__,
    path='/upload-vae',
    title='RaptGen: Upload VAE',
    name='Upload VAE'
)

def make_parameters_form() -> html.Div:
    return html.Div([
        html.P([
            dbc.Label(html.B("Model Name (REQUIRED)"), html_for="model_name"),
            dbc.Input(type="text", id="model_name", placeholder="Enter model name")
        ]),
        html.P([
            dbc.Col([
                dbc.Label("published date", html_for="published_date"),
            ], width=10),
            dcc.DatePickerSingle(
                date = datetime.datetime.today().date(),
                display_format = "Y/M/D",
                id = "published_date",
            )
        ]),
        html.P([
            dbc.Label("experiment", html_for="form_experiment"),
            dbc.Input(type="text", id="form_experiment", placeholder="Enter an experiment name"),
        ]),
        html.P([
            dbc.Label("round", html_for="form_round"),
            dbc.Input(type="text", id="form_round", placeholder="Enter round"),
        ]),
        dbc.Row([
            dbc.Col(
                html.P([
                    dbc.Label("Forward adapter", html_for="form_fwd_adapter"),
                    dbc.Input(type="text", id="form_fwd_adapter", placeholder="Enter the sequence of forward adapter"),
                ]),
            ),
            dbc.Col(
                html.P([
                    dbc.Label("Backward adapter", html_for="form_rev_adapter"),
                    dbc.Input(type="text", id="form_rev_adapter", placeholder="Enter the sequence of backward adapter"),
                ]),
            ),
            dbc.Col(
                html.P([
                    dbc.Button(
                        "Estimate Adapters", id="btn_estimate_adapters", className="btn btn-primary", disabled=True
                    )
                ])
            , width="auto", align="end"),
        ]),
        dbc.Row([
            dbc.Col(
                html.P([
                    dbc.Label("Target Length (REQUIRED)", html_for="form_target_length"),
                    dbc.Input(type="number", id="form_target_length", placeholder="Enter the total length of an sequence"),
                ]),
            ),
            dbc.Col(
                html.P([
                    dbc.Button(
                        "Estimate Target Length", id="btn_estimate_target_length", className="btn btn-primary", disabled=True
                    )
                ])
            , width="auto", align="end"),
        ]),
        html.P([
            dbc.Label("Filtering tolerance", html_for="form_tolerance"),
            dbc.Input(type="number", id="form_tolerance", placeholder="Enter filtering tolerance"),
        ]),
        html.P([
            dbc.Label("minimum count", html_for="form_mcount"),
            dbc.Input(type="number", id="form_mcount", placeholder="Enter "),
        ]),
        html.P([
            dbc.Label("Epochs", html_for="form_epochs"),
            dbc.Input(type="number", id="form_epochs", placeholder="Enter the maximum number of epochs"),
        ]),
        html.P([
            dbc.Label("Beta weighting epochs", html_for="form_beta_epochs"),
            dbc.Input(type="number", id="form_beta_epochs", placeholder="Enter the number of epochs for beta weighting"),
        ]),
        html.P([
            dbc.Label("Match forcing epochs", html_for="form_match_epochs"),
            dbc.Input(type="number", id="form_match_epochs", placeholder="Enter the number of epochs under the match forcing regularization"),
        ]),
        html.P([
            dbc.Label("Match Cost", html_for="form_match_cost"),
            dbc.Input(type="number", id="form_match_cost", placeholder="Enter the cost value of match forcing regularization"),
        ]),
        html.P([
            dbc.Label("early stopping epochs", html_for="form_earlystop"),
            dbc.Input(type="number", id="form_earlystop", placeholder="Enter the early stopping epochs"),
        ]),
        html.P([
            dbc.Label("CUDA num_workers", html_for="form_worker"),
            dbc.Input(type="number", id="form_worker", placeholder="Enter the number of workers for CUDA"),
        ]),
        html.P([
            dbc.Label("CUDA pin_memory", html_for="dropdown_pinm"),
            dcc.Dropdown(options=["True", "False"], id="dropdown_pinm"),
        ]),
        html.P([
            dbc.Label("VAE seed", html_for="form_VAEseed"),
            dbc.Input(type="number", id="form_VAEseed", placeholder="Enter the seed value used for constructing VAE"),
        ]),
    ], className="form-group", id="parameters-form")


def make_layout_upload() -> html.Div:
    return html.Div([
        html.H3("VAE model"),
        html.P([
            dcc.Loading([
                util.make_uploader("uploader-VAE"),
            ], id="uploader-VAE-wrapper"),
        ]),
        html.H3("HT-SELEX file"),
        html.P([
            dcc.Loading([
                util.make_uploader("uploader-SELEX"),
            ], id="uploader-SELEX-wrapper"),
        ]),
        html.H3("Parameters"),
        make_parameters_form(),
        html.P([
            dbc.Button("Next", id="next_btn", className="btn btn-lg btn-primary", disabled=True),
        ], className="d-grid gap-2 col-4 mx-auto"),
        html.Div(id="warning_label"),
    ])

def make_layout_encode(
    profile_df: pd.DataFrame,
    unique_df: pd.DataFrame,
) -> html.Div:
    return html.Div([
        html.Div(id="encode-interval-wrapper"),
        html.H3("Uploaded Parameters"),
        html.Div(
            dbc.Table.from_dataframe(
                profile_df.T,
                striped=True, bordered=True, hover=True, index=True
            ),
            id="profile-dataframe-table-wrapper",
            style = {"overflow": "scroll"}
        ),
        html.H3("Uploaded SELEX File (head 10 sequence)"),
        html.Div(
            dbc.Table.from_dataframe(
                unique_df.head(10),
                striped=True, bordered=True, hover=True
            ),
            id="unique-seq-dataframe-table-wrapper",
            style = {"overflow": "scroll"}
        ),
        dbc.Collapse([
            html.P([
                dbc.Button("Encode SELEX data", id="generate-coords-btn", className="btn btn-lg btn-primary"),
            ], className="d-grid gap-2 col-4 mx-auto"),
        ], id="generate-coords-btn-collapse", is_open=True),
        dbc.Collapse([
                dbc.Progress(id="encode-progress-bar", animated=True, className="mb-3"),
                html.P(id="encode-progress-bar-label"),
        ], id = "encode-collapse"),
        dbc.Collapse([
            html.P(id="encode-graph-wrapper"),
            html.P([
                dbc.Button("submit", id="submit-btn", className="btn btn-lg btn-primary"),
            ], id="submit-btn-wrapper", className="d-grid gap-2 col-4 mx-auto"),
            html.P(id="submit-notifier-wrapper"),
        ], id = "submit-collapse")
    ])

@callback(
    [
        Output('uploader-VAE-wrapper', 'children'),
        Output('VAE-statedict-base64', 'data'),
        Output('VAE-statedict-base64-exists', 'data'),
    ],
    Input('uploader-VAE', 'contents'),
    State('uploader-VAE', 'filename'),
    prevent_initial_call=True,
)
def validate_VAE_model(contents: str, filename: str):
    _, content_base64 = contents.split(",")

    if not filename.endswith(".mdl"):
        return (
            [
                util.make_uploader("uploader-VAE"),
                html.Div([
                    html.H5("Warning: Invalid file uploaded."),
                    html.B("please upload \".mdl\" file."),
                ], className="alert alert-dismissible alert-warning")
            ],
            "",
            False,
        )
    
    try:
        state_dict = util.CPU_Unpickler(io.BytesIO(base64.b64decode(content_base64))).load()
        motif_len = int(state_dict["decoder.emission.2.weight"].shape[0] / 4)
        embed_size = state_dict["decoder.fc1.0.weight"].shape[1]

        model = CNN_PHMM_VAE(
            motif_len = motif_len, 
            embed_size = embed_size,
        )
        model.load_state_dict(state_dict)

    except KeyError:
        return (
            [
                util.make_uploader("uploader-VAE"),
                html.Div([
                    html.H5("Warning: Invalid file uploaded."),
                    html.B("uploaded file does not match CNN_PHMM_VAE model")
                ], className="alert alert-dismissible alert-warning")
            ],
            "",
            False,
        )
    except pickle.UnpicklingError as e:
        return (
            [
                util.make_uploader("uploader-VAE"),
                html.Div([
                    html.H5("Warning: Invalid file uploaded."),
                    html.B("can't unpack uploaded file. please input pickled file"),
                    html.P([
                        html.P(string)
                        for string in f"{e}".split("\n")
                    ]),
                ], className="alert alert-dismissible alert-warning")
            ],
            "",
            False,
        )
    except Exception as e:
        return (
            [
                util.make_uploader("uploader-VAE"),
                html.Div([
                    html.H5("Warning: Invalid file uploaded."),
                    html.P([
                        html.P(string)
                        for string in f"{e}".split("\n")
                    ]),
                ], className="alert alert-dismissible alert-warning")
            ],
            "",
            False,
        )
    else:
        return (
            [
                html.Div([
                    html.H5("VAE model has been successfully uploaded!"),
                    html.P(f"filename: {filename}"),
                    html.P(f"motif length: {motif_len}"),
                    html.P(f"embedded size: {embed_size}"),
                ], className="alert alert-dismissible alert-success")
            ],
            content_base64,
            True,
        )

@callback(
    [
        Output('uploader-SELEX-wrapper', 'children'),
        Output('SELEX-dataframe-json', 'data'),
        Output('SELEX-dataframe-json-exists', 'data'),
    ],
    Input('uploader-SELEX', 'contents'),
    State('uploader-SELEX', 'filename'),
    prevent_initial_call=True,
)
def validate_SELEX_file(contents: str, filename: str):
    _, content_base64 = contents.split(",")

    if not (filename.endswith((".fasta", ".fastq", ".csv"))):
        return (
            [
                util.make_uploader("uploader-SELEX"),
                html.Div([
                    html.H5("Warning: Invalid file uploaded."),
                    html.B("please upload \".csv\", \".fasta\" or \".fastq\" file."),
                ], className="alert alert-dismissible alert-warning")
            ],
            "",
            False,
        )
    
    try:
        with tempfile.NamedTemporaryFile(mode="w+") as f:
            textio = io.TextIOWrapper(
                io.BytesIO(base64.b64decode(content_base64)),
                encoding="utf-8"
            )
            f.write(textio.read())
            f.flush()
            if filename.endswith(".fasta"):
                selex_df = read_SELEX_data(filepath=f.name, filetype="fasta")
            elif filename.endswith(".fastq"):
                selex_df = read_SELEX_data(filepath=f.name, filetype="fastq")
            else:
                selex_df: pd.DataFrame = pd.read_csv(filepath=f.name)
                if not "ID" in selex_df.columns:
                    selex_df["ID"] = [f"seq_{i}" for i in range(len(selex_df))]
                if "Sequence" in selex_df.columns:
                    invalid_df = selex_df[selex_df["Sequence"].str.match("[AUCGT]*") == False]
                    if len(invalid_df) > 0:
                        raise Exception(
                            "\n".join(["invalid values found"] + [
                                f"Index:{index}, ID: {row['ID']}, Sequence: {row['Sequence']}"
                                for index, row in invalid_df.iterrows()
                            ])
                        )
                else:
                    raise Exception("\"Sequence\" column not found on the input csv")
            
    except Exception as e:
        return (
            [
                util.make_uploader("uploader-SELEX"),
                html.Div([
                    html.H5("Warning: Failed to parse file."),
                    html.P([
                        html.P(string)
                        for string in f"{e}".split("\n")
                    ]),
                ], className="alert alert-dismissible alert-warning")
            ],
            "",
            False,
        )
    
    else:
        return (
            [
                html.Div([
                    html.H5("SELEX file has been successfully uploaded!"),
                    html.P(f"filename: {filename}"),
                ], className="alert alert-dismissible alert-success")
            ],
            selex_df.to_json(),
            True,
        )

@callback(
    [
        Output("form_fwd_adapter", "invalid"),
        Output("form_rev_adapter", "invalid"),
    ],
    [
        Input("form_fwd_adapter", "value"),
        Input("form_rev_adapter", "value"),
    ],
    prevent_initial_call=True,
)
def validate_adapters(fwd_adapter, rev_adapter):
    if fwd_adapter == None:
        fwd_adapter = ""
    if rev_adapter == None:
        rev_adapter = ""
    fwd_flag = re.fullmatch("[ATCGU]*", fwd_adapter) != None
    rev_flag = re.fullmatch("[ATCGU]*", rev_adapter) != None
    return not fwd_flag, not rev_flag

@callback(
    [
        Output("btn_estimate_adapters", "disabled"),
        Output("btn_estimate_target_length", "disabled"),
    ],
    # Input('uploader-SELEX', 'contents'),
    Input('SELEX-dataframe-json-exists', 'data'),
    Input('form_target_length', 'value'),
    prevent_initial_call=True,
)
def activate_estimation_btns(SELEX_exists: Union[None, bool], target_length: Union[None, int]):
    if SELEX_exists in [None, False]: return (True, True)
    elif target_length == None: return (True, False)
    else: return (False, False)

@callback(
    Output("form_target_length", "value"),
    Input("btn_estimate_target_length", "n_clicks"),
    State("SELEX-dataframe-json", "data"),
    prevent_initial_call=True,
)
def estimate_target_length(_, SELEX_dataframe_json: str):
    df: pd.DataFrame = pd.read_json(io.StringIO(SELEX_dataframe_json))
    unique_reads, _ = unify_duplicates(df["Sequence"].to_list())
    estimated_target_length = calc_target_length(unique_reads)
    return estimated_target_length

@callback(
    [
        Output("form_fwd_adapter", "value"),
        Output("form_rev_adapter", "value"),
        # Output("form_target_length", "value"),
    ],
    Input("btn_estimate_adapters", "n_clicks"),
    State("SELEX-dataframe-json", "data"),
    State("form_target_length", "value"),
    prevent_initial_call=True,
)
def estimate_adapters_wrapper(_, SELEX_dataframe_json: str, target_length: Union[None, int]):
    df: pd.DataFrame = pd.read_json(io.StringIO(SELEX_dataframe_json))
    unique_reads, _ = unify_duplicates(df["Sequence"].to_list())

    if target_length == None:
        target_length = calc_target_length(unique_reads)
    
    fwd_adapter, rev_adapter = estimate_adapters(unique_reads, target_length)
    
    return (fwd_adapter, rev_adapter)

@callback(
    Output("next_btn", "disabled"),
    [
        Input("VAE-statedict-base64-exists", "data"),
        Input("SELEX-dataframe-json-exists", "data"),
        Input("model_name", "value"),
        Input("form_target_length", "value"),
        Input("form_fwd_adapter", "invalid"),
        Input("form_rev_adapter", "invalid"),
    ],
    prevent_initial_call=True,
)
def activate_submit_btn(
    VAE_statedict_base64_exists, 
    SELEX_dataframe_json_exists, 
    model_name,
    target_length,
    fwd_adapter_is_invalid,
    rev_adapter_is_invalid,
):
    if VAE_statedict_base64_exists == True \
        and SELEX_dataframe_json_exists == True\
        and model_name not in [None, ""] \
        and target_length != None \
        and fwd_adapter_is_invalid == False \
        and rev_adapter_is_invalid == False:
        return False
    else: return True


@callback(
    [
        # Output("next_btn", "disabled"),
        Output("profile-dataframe-json", "data"),
        Output("unique-seq-dataframe-json", "data"),
        Output("vu-page", "children"),
    ],
    Input("next_btn", "n_clicks"),

    State("model_name", "value"),
    State("published_date", "date"),
    State("form_experiment", "value"),
    State("form_round", "value"),
    State("form_fwd_adapter", "value"),
    State("form_rev_adapter", "value"),
    State("form_target_length", "value"),
    # State("form_fs_length", "value"),
    State("form_tolerance", "value"),
    State("form_mcount", "value"),
    # State("form_embed_dim", "value"),
    State("form_epochs", "value"),
    State("form_beta_epochs", "value"),
    State("form_match_epochs", "value"),
    State("form_match_cost", "value"),
    State("form_earlystop", "value"),
    State("form_worker", "value"),
    State("dropdown_pinm", "value"),
    # State("form_modellen", "value"),
    State("form_VAEseed", "value"),

    State("VAE-statedict-base64", "data"),
    State("SELEX-dataframe-json", "data"),
    prevent_initial_call=True,
)
def process_df_csv(
    _,

    model_name,
    published_date,
    experiment_name,
    round,
    fwd_adapter,
    rev_adapter,
    target_length,
    # filtered_standard_length,
    tolerance,
    mcount,
    # embed_dim,
    epochs,
    beta_epochs,
    match_epochs,
    match_cost,
    earlystop,
    worker,
    pin_memory,
    # model_length,
    seed,

    VAE_statedict_base64,
    SELEX_dataframe_json,
):

    state_dict = util.CPU_Unpickler(io.BytesIO(base64.b64decode(VAE_statedict_base64))).load()
    motif_len = int(state_dict["decoder.emission.2.weight"].shape[0] / 4)
    embed_size = state_dict["decoder.fc1.0.weight"].shape[1]

    model = CNN_PHMM_VAE(
        motif_len = motif_len, 
        embed_size = embed_size,
    )
    model.load_state_dict(state_dict)

    filtered_standard_length = target_length - len(fwd_adapter) - len(rev_adapter)

    record = {
        "published_time": date.fromisoformat(published_date).strftime("%Y/%m/%d"),
        "experiment": experiment_name,
        "round": round,
        "fwd_adapter": fwd_adapter,
        "rev_adapter": rev_adapter,
        "target_length": target_length,
        "filtering_standard_length": filtered_standard_length,
        "filtering_tolerance": tolerance,
        "filtering_method": "default",
        "minimum_count": mcount,
        "embedding_dim": embed_size,
        "epochs": epochs,
        "beta_weight_epochs": beta_epochs,
        "match_forcing_epochs": match_epochs,
        "match_cost": match_cost,
        "early_stopping_epochs": earlystop,
        "CUDA_num_workers": worker,
        "CUDA_pin_memory": True if pin_memory == "True" else False,
        "pHMM_VAE_model_length": motif_len,
        "pHMM_VAE_seed": seed
    }

    profile_df = pd.DataFrame.from_records([record])
    
    profile_df.rename(index={0: model_name}, inplace=True)

    # generate unique seq dataframe
    SELEX_df: pd.DataFrame = pd.read_json(io.StringIO(SELEX_dataframe_json))

    SELEX_df = SELEX_df[SELEX_df['Sequence'].apply(
        lambda seq: default_filterfunc(
            seq,
            fwd_adapter = fwd_adapter,
            rev_adapter = rev_adapter,
            target_length = target_length,
            # tolerance = target_length-len(fwd_adapter)-len(rev_adapter)-1,
            tolerance = 0
        )
    )] # filter with t_len ± tolerance

    sequences, duplicates = unify_duplicates(SELEX_df['Sequence'].to_list())
    unique_df = pd.DataFrame({
        "Sequence": sequences,
        "Duplicates": duplicates,
    })

    unique_df["Without_Adapters"] = unique_df["Sequence"].apply(
        lambda seq: default_cutfunc(
            read = seq,
            fwd_adapter = fwd_adapter,
            rev_adapter = rev_adapter,
        )
    )

    return [
        profile_df.to_json(),
        unique_df.to_json(),
        make_layout_encode(
            profile_df = profile_df,
            unique_df = unique_df,
        )
    ]

@callback(
    Output("encode-interval-wrapper", "children"),
    Input("generate-coords-btn", "n_clicks"),
    State("unique-seq-dataframe-json", "data"),
    State("VAE-statedict-base64", "data"),
    prevent_initial_call=True,
)
def encode_SELEX_wrapper(n_clicks, unique_seq_dataframe_json, VAE_statedict_base64):
    state_dict = util.CPU_Unpickler(io.BytesIO(base64.b64decode(VAE_statedict_base64))).load()
    motif_len = int(state_dict["decoder.emission.2.weight"].shape[0] / 4)
    embed_size = state_dict["decoder.fc1.0.weight"].shape[1]

    model = CNN_PHMM_VAE(
        motif_len = motif_len, 
        embed_size = embed_size,
    )
    model.load_state_dict(state_dict)

    unique_seq_df: pd.DataFrame = pd.read_json(io.StringIO(unique_seq_dataframe_json))

    global ms_generator
    ms_generator = MessageBroker(
        id = 42,
        target = encode_SELEX
    )
    ms_generator.run(
        dataframe = unique_seq_df,
        model = model,
        num_split = 20
    )

    return dcc.Interval(id="encode-interval", n_intervals=100)

def encode_SELEX(
    set_message: Callable, 
    notify_end: Callable, 
    dataframe: pd.DataFrame, 
    model: CNN_PHMM_VAE,
    num_split: int,
):
    data_list = dataframe["Without_Adapters"].to_list()
    num_all = len(data_list)
    num_chunk = int(num_all / num_split)
    num_total = 0
    coords_x_list = list()
    coords_y_list = list()

    set_message("0")

    for i in range(0, len(data_list), num_chunk):
        chunk = data_list[i : i+num_chunk]
        if len(chunk) == 0:
            break

        coords_x, coords_y = np.array(embed_sequences(
            sequences = chunk,
            model = model,
        )).T.tolist()
        coords_x_list.append(coords_x)
        coords_y_list.append(coords_y)
        
        num_total += len(chunk)
        set_message(str(num_total/num_all * 100))
    

    coords_x_all = np.concatenate(coords_x_list)
    coords_y_all = np.concatenate(coords_y_list)

    dataframe["coord_x"] = coords_x_all
    dataframe["coord_y"] = coords_y_all
        
    notify_end(dataframe)
    return


@callback(
    [
        Output("encode-progress-bar", "value"),
        Output("encode-progress-bar-label", "children"),
        Output("encode-collapse", "is_open"),
        Output("generate-coords-btn-collapse", "is_open"),
        Output("now-encoding-flag", "data"),
    ],
    Input("encode-interval", "n_intervals"),
    prevent_initial_call=True,
)
def fetch_progress(n_progress):
    header = "Encoding SELEX data... "
    global ms_generator
    if ms_generator == None:
        return [
            0, header + "0 %", False, True, False
        ]

    if ms_generator.fetch_started():
        if ms_generator.fetch_finished():
            return [
                100, header + "100 %", False, False, False
            ]
        else:
            value = float(ms_generator.get_message())
            return [
                value, header + f"{value:.1f} %", True, False, True
            ]
    else:
        return [
            0, header + "0 %", False, True, False
        ]
    
@callback(
    # Output("encode-interval-wrapper", "children"),
    Output("encode-interval", "disabled"),
    Input("now-encoding-flag", "data"),
    prevent_initial_call=True,
)
def silence_interval(now_encoding_flag):
    if now_encoding_flag == False:
        return True
    else:
        return False

@callback(
    [
        Output("encode-graph-wrapper", "children"),
        Output("submit-collapse", "is_open"),
    ],
    Input("now-encoding-flag", "data"),
    prevent_initial_call=True,
)
def process_finish(now_encoding_flag):
    if now_encoding_flag == True:
        return "", False
    
    submit_seq_df: pd.DataFrame = ms_generator.get_result()

    fig = go.Figure(
        layout = dict(
            height = 800,
            title = "preview",
            template = "ggplot2",
            yaxis = dict(
                scaleanchor = "x"
            )
        )
    )
    fig.update_xaxes(range=[-3.5, 3.5])
    fig.update_yaxes(range=[-3.5, 3.5])

    plot.plot_SELEX(
        fig = fig,
        df_SELEX_data = submit_seq_df,
        color = "silver",
    )

    return dcc.Graph(figure=fig, id="preview-figure"), True
    

@callback(
    [
        Output("submit-notifier-wrapper", "children"),
        Output("submit-btn-wrapper", "children"),
    ],
    Input("submit-btn", "n_clicks"),
    State("profile-dataframe-json", "data"),
    State("VAE-statedict-base64", "data"),
    prevent_initial_call=True,
)
def submit(n_clicks, profile_dataframe_json, VAE_statedict_base64):
    SELEX_df: pd.DataFrame = ms_generator.get_result()
    profile_df = pd.read_json(io.StringIO(profile_dataframe_json)) 

    state_dict = util.CPU_Unpickler(io.BytesIO(base64.b64decode(VAE_statedict_base64))).load()

    profile_df_final = pd.concat([util.get_profile_df(), profile_df])

    with open(f"{util.get_data_dirname()}/profile_dataframe.pkl", "wb") as f:
        pickle.dump(profile_df_final, f)

    savepath = Path(f"{util.get_data_dirname()}/items/{profile_df.index[0]}")
    savepath.mkdir()

    SELEX_df.to_pickle(f"{savepath}/unique_seq_dataframe.pkl")
    
    with open(f"{savepath}/VAE_model.pkl", "wb") as f:
        pickle.dump(state_dict, f)

    return (
        html.Div([
            html.H5("The VAE model has been successfully submitted"),
        ], className="alert alert-dismissible alert-success"),
        "",
    )
            
storage: html.Div = html.Div([
    dcc.Store(id="now-encoding-flag", data=False),
    dcc.Store(id="VAE-statedict-base64"),
    dcc.Store(id="VAE-statedict-base64-exists"),
    dcc.Store(id="SELEX-dataframe-json"),
    dcc.Store(id="SELEX-dataframe-json-exists"),
    dcc.Store(id="profile-dataframe-json"),
    dcc.Store(id="unique-seq-dataframe-json"),
])

ms_generator: Union[None, MessageBroker] = None

layout = dbc.Container([
    storage,
    html.H1("Upload VAE"),
    html.Hr(),
    html.Div(make_layout_upload(), id="vu-page")
])
