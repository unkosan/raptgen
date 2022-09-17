from typing import Any, Dict, List
from raptgen.core.algorithms import *
from raptgen.core.preprocessing import *
from raptgen.core.train import *
import pandas as pd
from raptgen.core.algorithms import CNN_PHMM_VAE
import dash_bootstrap_components as dbc
import os

from . import util

from dash import html, dcc, Input, Output, State, callback
import dash

dash.register_page(
    __name__,
    path='/delete-data',
    title='RaptGen: Remove Data',
    name='Remove data'
)

def make_layout() -> html.Div:
    return html.Div([
        html.H3("Delete VAE model"),
        html.P(dbc.Row([
            dbc.Col([
                html.Label("Select VAE model to delete"),
                dcc.Dropdown(
                    id="de-VAE-model-name",
                    options=[
                        {"label": entry, "value": entry}
                        for entry in util.get_profile_df().index
                    ] + [
                        {"label": "None", "value": "None"}
                    ],
                    value="None",
                )
            ]),
            dbc.Col([
                dbc.Button("Delete", id="de-delete-VAE-btn", disabled=True)
            ], width="auto", align="end")
        ]), id="de-VAE-deletion-wrapper"),

        html.H3("Delete GMM model"),
        html.P([
            html.P([
                html.Label("Select VAE model"),
                dcc.Dropdown(
                    id="de-GMM-deletion-VAE-model-name",
                    options=[
                        {"label": entry, "value": entry}
                        for entry in util.get_profile_df().index
                    ] + [
                        {"label": "None", "value": "None"}
                    ],
                    value="None",
                ),
            ]),
            html.P(dbc.Row([
                dbc.Col([
                    html.Label("Select GMM model to delete"),
                    dcc.Dropdown(
                        id="de-GMM-model-name",
                        options=[
                            {"label": "None", "value": "None"}
                        ],
                        value="None",
                    )
                ]),
                dbc.Col([
                    dbc.Button("Delete", id="de-delete-GMM-btn", disabled=True)
                ], width="auto", align="end")
            ])),
        ], id="de-GMM-deletion-wrapper"),

        html.H3("Delete measured data"),
        html.P(dbc.Row([
            dbc.Col([
                html.Label("Select Measured Data to delete"),
                dcc.Dropdown(
                    id="de-measured-data-name",
                    options=[
                        {"label": entry, "value": entry}
                        for entry in util.get_measured_value_names()
                    ] + [
                        {"label": "None", "value": "None"}
                    ],
                    value="None",
                )
            ]),
            dbc.Col([
                dbc.Button("Delete", id="de-delete-measured-data-btn", disabled=True)
            ], width="auto", align="end")
        ]), id="de-measured-data-deletion-wrapper"),
    ])
    

@callback(
    [
        Output("de-GMM-model-name", "options"),
        Output("de-GMM-model-name", "value"),
    ],
    Input("de-GMM-deletion-VAE-model-name", "value"),
)
def update_GMM_dropdown(VAE_model_name):
    options = [
        {"label": "None", "value": "None"}
    ]

    if VAE_model_name not in ["None", None]:
        GMM_df_path = Path(f"{util.get_data_dirname()}/items/{VAE_model_name}/best_gmm_dataframe.pkl")
        if GMM_df_path.exists():
            df_GMM: pd.DataFrame = pd.read_pickle(str(GMM_df_path))
            options += [
                {"label": entry, "value": entry}
                for entry in df_GMM.index
            ]

    return options, "None"

@callback(
    Output("de-delete-VAE-btn", "disabled"),
    Input("de-VAE-model-name", "value"),
    prevent_initial_call=True,
)
def activate_VAE_deletion_btn(VAE_model_name):
    if VAE_model_name in [None, "None"]:
        return True
    else:
        return False

@callback(
    Output("de-delete-GMM-btn", "disabled"),
    Input("de-GMM-model-name", "value"),
    prevent_initial_call=True,
)
def activate_GMM_deletion_btn(GMM_model_name):
    if GMM_model_name in [None, "None"]:
        return True
    else:
        return False

@callback(
    Output("de-delete-measured-data-btn", "disabled"),
    Input("de-measured-data-name", "value"),
    prevent_initial_call=True,
)
def activate_measured_data_deletion_btn(measured_data_name):
    if measured_data_name in [None, "None"]:
        return True
    else:
        return False

@callback(
    Output("de-VAE-deletion-wrapper", "children"),
    Input("de-delete-VAE-btn", "n_clicks"),
    State("de-VAE-model-name", "value"),
    prevent_initial_call=True,
)
def delete_VAE(n_clicks, VAE_model_name):
    profile_df = util.get_profile_df()
    profile_df.drop(VAE_model_name, inplace=True)
    profile_df.to_pickle(f"{util.get_data_dirname()}/profile_dataframe.pkl")

    VAE_dir = Path(f"{util.get_data_dirname()}/items/{VAE_model_name}")
    for child in VAE_dir.glob('*'):
        child.unlink(missing_ok=True)
    VAE_dir.rmdir()

    return (
        html.Div([
            html.H5("Successfully deleted")
        ], className="alert alert-dismissible alert-success"),
    )

@callback(
    Output("de-GMM-deletion-wrapper", "children"),
    Input("de-delete-GMM-btn", "n_clicks"),
    State("de-GMM-deletion-VAE-model-name", "value"),
    State("de-GMM-model-name", "value"),
    prevent_initial_call=True,
)
def delete_GMM(n_clicks, VAE_model_name, GMM_model_name):
    GMM_df = util.get_gmm_dataframe(VAE_model_name)
    GMM_df.drop(GMM_model_name, inplace=True)
    GMM_df.to_pickle(f"{util.get_data_dirname()}/items/{VAE_model_name}/best_gmm_dataframe.pkl")

    return (
        html.Div([
            html.H5("Successfully deleted")
        ], className="alert alert-dismissible alert-success"),
    )

@callback(
    Output("de-measured-data-deletion-wrapper", "children"),
    Input("de-delete-measured-data-btn", "n_clicks"),
    State("de-measured-data-name", "value"),
    prevent_initial_call=True,
)
def delete_measured_data(n_clicks, measured_data_name):
    Path(f"{util.get_data_dirname()}/measured_values/{measured_data_name}").unlink()

    return (
        html.Div([
            html.H5("Successfully deleted")
        ], className="alert alert-dismissible alert-success"),
    )

layout = dbc.Container([
    dcc.Location(id='de-url', refresh=True),
    html.H1("Remove Data"),
    html.Hr(),
    html.Div(make_layout(), id="de-page")
])

@callback(
    Output('de-page', 'children'),
    Input('de-url', 'href'),
)
def refresh_page(url):
    return [make_layout()]
