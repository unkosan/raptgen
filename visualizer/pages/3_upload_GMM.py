import io
from typing import Any, Dict, List
from raptgen.core.algorithms import *
from raptgen.core.preprocessing import *
from raptgen.core.train import *
import pandas as pd
import pickle
from raptgen.core.algorithms import CNN_PHMM_VAE
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import base64

from . import util
from . import plot

import dash
from dash import html, dcc, Input, Output, State, callback

dash.register_page(
    __name__,
    path='/upload-gmm',
    title='RaptGen: Upload GMM',
    name='Upload GMM'
)

def make_layout() -> html.Div:
    return html.Div([
        html.H3("VAE model"),
        html.P(
            dcc.Dropdown(
                id="gu-VAE-model-name",
                options=[
                    {"label": entry, "value": entry}
                    for entry in util.get_profile_df().index
                ],
            ),
        ),
        html.H3("GMM model"),
        html.P([
            dcc.Loading([
                util.make_uploader("gu-uploader-GMM")
            ], id="gu-uploader-GMM-wrapper"),
        ]),
        html.H3("Parameters"),
        make_parameters_form(),
        dbc.Collapse([
            html.P([
                dbc.Button("Next", id="gu-next-btn", className="btn btn-lg btn-primary", disabled=True),
            ], className="d-grid gap-2 col-4 mx-auto"),
        ], id="gu-next-btn-collapse", is_open=True),
        dbc.Collapse([
            html.H3("Preview"),
            html.P(id="gu-preview-table"),
            dcc.Graph(id="gu-preview-figure"),
            html.P([
                dbc.Button("submit", id="gu-submit-btn", className="btn btn-lg btn-primary"),
            ], id="gu-submit-btn-wrapper", className="d-grid gap-2 col-4 mx-auto"),
            html.P(id="gu-submit-notifier-wrapper"),
        ], id="gu-submit-collapse"),
    ])

def make_parameters_form() -> html.Div:
    return html.Div([
        html.P([
            dbc.Label(html.B("Model Name (REQUIRED)"), html_for="gu-model-name"),
            dbc.Input(type="text", id="gu-model-name", placeholder="Enter model name")
        ]),
        html.P([
            dbc.Label("seed", html_for="gu-form-seed"),
            dbc.Input(type="text", id="gu-form-seed", placeholder="seed")
        ]),
        html.P([
            dbc.Label("the type of GMM model", html_for="gu-model-type"),
            dbc.Input(type="text", id="gu-form-model-type", placeholder="model type form")
        ]),
    ], className="form-group", id="gu-parameters-form")


@callback(
    [
        Output("gu-uploader-GMM-wrapper", "children"),
        Output("gu-GMM-model-base64", "data"),
        Output("gu-GMM-model-base64-exists", "data")
    ],
    Input("gu-uploader-GMM", "contents"),
    State("gu-uploader-GMM", "filename"),
    prevent_initial_call=True,
)
def validate_GMM_model(contents: str, filename: str):
    _, content_base64 = contents.split(",")

    if not filename.endswith(".pkl"):
        return (
            [
                util.make_uploader("gu-uploader-GMM"),
                html.P([
                    html.H5("Warning: Invalid file uploaded."),
                    html.B("please upload \".pkl\" file."),
                ], className="alert alert-dismissible alert-warning")
            ],
            "",
            False,
        )
    
    try:
        GMM_model = pickle.load(io.BytesIO(base64.b64decode(content_base64)))
        if type(GMM_model) not in [GaussianMixture, BayesianGaussianMixture]:
            return (
                [
                    util.make_uploader("gu-uploader-GMM"),
                    html.P([
                        html.H5("Warning: Invalid file uploaded."),
                        html.B("uploaded file does not match Gaussaian Mixture model")
                    ], className="alert alert-dismissible alert-warning")
                ],
                "",
                False,
            )

    except pickle.UnpicklingError as e:
        return (
            [
                util.make_uploader("gu-uploader-GMM"),
                html.P([
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
                util.make_uploader("gu-uploader-GMM"),
                html.P([
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
                html.P([
                    html.H5("GMM model has been successfully uploaded!"),
                    html.P(f"filename: {filename}"),
                    html.P(f"GMM type: {type(GMM_model)}"),
                    html.P(f"num_components: {len(GMM_model.weights_)}")
                ], className="alert alert-dismissible alert-success")
            ],
            content_base64,
            True,
        )

@callback(
    Output("gu-next-btn", "disabled"),
    [
        Input("gu-model-name", "value"),
        Input("gu-VAE-model-name", "value"),
        Input("gu-GMM-model-base64-exists", "data"),
    ],
    prevent_initial_call=True,
)
def activate_next_btn(
    model_name: Union[None, str],
    VAE_model_name: Union[None, str],
    GMM_model_base64_exists: Union[None, str]
):
    if GMM_model_base64_exists == True \
        and VAE_model_name != None \
        and model_name != None:
        return False
    else: return True


@callback(
    [
        Output("gu-preview-table", "children"),
        Output("gu-preview-figure", "figure"),
        Output("gu-GMM-dataframe-json", "data"),
        Output("gu-staged-GMM-model-base64", "data"),
        Output("gu-submit-collapse", "is_open"),
        Output("gu-staged-VAE-model-name", "data"),
    ],
    Input("gu-next-btn", "n_clicks"),
    State("gu-GMM-model-base64", "data"),
    State("gu-VAE-model-name", "value"),

    State("gu-model-name", "value"),
    State("gu-form-seed", "value"),
    State("gu-form-model-type", "value"),
    prevent_initial_call=True,
)
def preview(n_clicks, GMM_model_base64, VAE_model_name, model_name, seed, model_type):
    GMM_model = pickle.load(io.BytesIO(base64.b64decode(GMM_model_base64)))

    with open(f"{util.get_data_dirname()}/items/{VAE_model_name}/VAE_model.pkl", "rb") as f:
        state_dict = util.CPU_Unpickler(f).load()
    
    motif_len = int(state_dict["decoder.emission.2.weight"].shape[0] / 4)
    embed_size = state_dict["decoder.fc1.0.weight"].shape[1]

    VAE_model = CNN_PHMM_VAE(
        motif_len = motif_len, 
        embed_size = embed_size,
    )
    VAE_model.load_state_dict(state_dict)
    VAE_model.eval()

    unique_df = pd.read_pickle(f"{util.get_data_dirname()}/items/{VAE_model_name}/unique_seq_dataframe.pkl")

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
        df_SELEX_data = unique_df,
        color = "silver",
    )

    GMM_params = {
        "IDs": np.array(range(len(GMM_model.means_))),
        "weights": GMM_model.weights_,
        "means": GMM_model.means_,
        "covariances": GMM_model.covariances_,
    }

    plot.plot_GMM(
        fig = fig,
        model = VAE_model,
        GMM_params = GMM_params,
        colorscale_name = "Viridis"
    )

    record = {
        "GMM_num_components": len(GMM_params["weights"]),
        "GMM_seed": seed,
        "GMM_optimal_model": None,
        "GMM_model_type": model_type,
    }

    GMM_dataframe: pd.DataFrame = pd.DataFrame.from_records([record])

    GMM_dataframe.rename(index={0: model_name}, inplace=True)
    GMM_dataframe.index.name = "name"

    GMM_binary = pickle.dumps(GMM_model)
    encoded_model = base64.b64encode(GMM_binary).decode()

    return (
        html.Div(
            dbc.Table.from_dataframe(
                GMM_dataframe.T,
                striped=True, bordered=True, hover=True, index=True
            ),
            id="gu-profile-dataframe-table-wrapper",
            style = {"overflow": "scroll"}
        ),
        fig,
        GMM_dataframe.to_json(),
        encoded_model,
        True,
        VAE_model_name,
    )

@callback(
    [
        Output("gu-submit-notifier-wrapper", "children"),
        Output("gu-submit-btn-wrapper", "children"),
        Output("gu-next-btn-collapse", "is_open")
    ],
    Input("gu-submit-btn", "n_clicks"),
    State("gu-GMM-dataframe-json", "data"),
    State("gu-staged-GMM-model-base64", "data"),
    State("gu-staged-VAE-model-name", "data"),
    prevent_initial_call=True,
)
def submit(n_clicks, GMM_dataframe_json: str, GMM_model_base64: str, VAE_model_name):
    GMM_df: pd.DataFrame = pd.read_json(io.StringIO(GMM_dataframe_json))
    GMM_model = pickle.loads(base64.b64decode(GMM_model_base64.encode()))
    
    model_name = GMM_df.index[0]
    GMM_df.at[model_name, "GMM_optimal_model"] = GMM_model

    GMM_df_final = pd.concat([util.get_gmm_dataframe(VAE_model_name), GMM_df])
    GMM_df_final.to_pickle(f"{util.get_data_dirname()}/items/{VAE_model_name}/best_gmm_dataframe.pkl")

    return (
        html.Div([
            html.H5("The GMM model has been successfully submitted"),
        ], className="alert alert-dismissible alert-success"),
        "",
        False
    )

storage: html.Div = html.Div([
    dcc.Store(id="gu-GMM-model-base64"),
    dcc.Store(id="gu-GMM-model-base64-exists"),
    dcc.Store(id="gu-staged-GMM-model-base64"),
    dcc.Store(id="gu-staged-VAE-model-name"),
    dcc.Store(id="gu-unique-seq-dataframe-json"),
    dcc.Store(id="gu-GMM-dataframe-json"),
])

layout = dbc.Container([
    storage,
    dcc.Location(id='gu-url', refresh=True),
    html.H1("Upload GMM"),
    html.Hr(),
    html.Div(make_layout(), id="gu-page")
])

@callback(
    Output('gu-page', 'children'),
    Input('gu-url', 'href'),
)
def refresh_page(url):
    return [make_layout()]