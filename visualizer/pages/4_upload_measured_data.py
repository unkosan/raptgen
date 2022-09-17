import io
from typing import Any, Dict, List
from raptgen.core.algorithms import *
from raptgen.core.preprocessing import *
from raptgen.core.train import *
import pandas as pd
import pickle
from raptgen.core.algorithms import CNN_PHMM_VAE
import dash_bootstrap_components as dbc

import dash
import plotly.graph_objects as go

import base64

from dash import html, dcc, Input, Output, State, callback
from . import util
from . import plot

dash.register_page(
    __name__,
    path='/upload-measured-data',
    title='RaptGen: Upload Measured Data',
    name="Upload Measured Data"
)

def make_layout() -> html.Div:
    return html.Div([
        html.H3("Measured Values"),
        html.P([
            dcc.Loading([
                util.make_uploader("mu-uploader-measured"),
            ], id="mu-uploader-measured-wrapper"),
        ]),
        dbc.Collapse([
            html.H3("Preview"),
            html.Div(id="mu-preview-table"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select VAE model to preview:")
                ], width="auto", align="center"),
                dbc.Col([
                    dcc.Dropdown(
                        id="mu-VAE-model-name",
                        options=[
                            {"label": entry, "value": entry}
                            for entry in util.get_profile_df().index
                        ] + [
                            {"label": "None", "value": "None"}
                        ],
                        value="None",
                    )
                ])
            ]),
            html.P(id="mu-figure-wrapper"),
        ], id="mu-preview-collapse"),
        html.P([
            dbc.Button("submit", id="mu-submit-btn", className="btn btn-lg btn-primary", disabled=True)
        ], id="mu-submit-btn-wrapper", className="d-grid gap-2 col-4 mx-auto"),
        html.P(id="mu-submit-notifier-wrapper")
    ])


@callback(
    [
        Output("mu-uploader-measured-wrapper", "children"),
        Output("mu-measured-dataframe-json", "data"),
        Output("mu-submit-btn", "disabled"),
        Output("mu-measured-file-name", "data"),
    ],
    Input("mu-uploader-measured", "contents"),
    State("mu-uploader-measured", "filename"),
    prevent_initial_call=True,
)
def validate_measured_table(contents: str, filename: str):
    _, content_base64 = contents.split(",")

    if not (filename.endswith((".pkl", ".csv"))):
        return (
            [
                util.make_uploader("mu-uploader-measured"),
                html.P([
                    html.H5("Warning: Invalid file uploaded."),
                    html.B("please upload \".pkl\" or \".csv\" file."),
                ], className="alert alert-dismissible alert-warning")
            ],
            "",
            True,
            None,
        )
    
    try:
        if filename.endswith(".pkl"):
            uploaded_object = pickle.loads(base64.b64decode(content_base64))
            if type(uploaded_object) != pd.DataFrame:
                return (
                    [
                        util.make_uploader("mu-uploader-measured"),
                        html.Div([
                            html.H5("Warning: Invalid file uploaded."),
                            html.B("uploaded file does not match pandas dataframe.")
                        ], className="alert alert-dismissible alert-warning"),
                    ],
                    "",
                    True,
                    None,
                )
        else:
            uploaded_object = pd.read_csv(io.BytesIO(base64.b64decode(content_base64)))
        
        uploaded_df: pd.DataFrame = uploaded_object

        assert "hue" in uploaded_df, "Hue (Series title) needed"
        assert "Sequence" in uploaded_df, "Sequence needed"
        assert "ID" in uploaded_df, "Sequence ID needed"
    
    except pickle.UnpicklingError as e:
        return (
            [
                util.make_uploader("mu-uploader-measured"),
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
            True,
            None,
        )
    except AssertionError as e:
        return (
            [
                util.make_uploader("mu-uploader-measured"),
                html.Div([
                    html.H5("Warning: The input file does not match the condition."),
                    html.B("The input file must have columns named \"hue\" \"ID\" \"Sequence\"."),
                    html.P([
                        html.P(string)
                        for string in f"{e}".split("\n")
                    ]),
                ], className="alert alert-dismissible alert-warning")
            ],
            "",
            True,
            None,
        )
    except Exception as e:
        return (
            [
                util.make_uploader("mu-uploader-measured"),
                html.Div([
                    html.H5("Warning: Invalid file uploaded."),
                    html.P([
                        html.P(string)
                        for string in f"{e}".split("\n")
                    ]),
                ], className="alert alert-dismissible alert-warning")
            ],
            "",
            True,
            None,
        )
    else:
        return (
            [
                html.Div([
                    html.H5("Measured values table has been successfully uploaded"),
                    html.P(f"filename: {filename}"),
                ], className="alert alert-dismissible alert-success")
            ],
            uploaded_df.to_json(),
            False,
            filename,
        )
        
@callback(
    [
        Output("mu-preview-table", "children"),
        Output("mu-preview-collapse", "is_open"),
    ],
    Input("mu-measured-dataframe-json", "data"),
    prevent_initial_call=True,
)
def preview(measured_dataframe_json: str):
    if measured_dataframe_json in [None, ""]:
        return "", False
    
    measured_df: pd.DataFrame = pd.read_json(measured_dataframe_json)

    return (
        html.Div(
            dbc.Table.from_dataframe(
                measured_df.head(),
                striped=True, bordered=True, hover=True, index=True
            ),
            style = {"overflow": "scroll"}
        ),
        True,
    )
    
@callback(
    Output("mu-figure-wrapper", "children"),
    Input("mu-VAE-model-name", "value"),
    State("mu-measured-dataframe-json", "data"),
    prevent_initial_call=True,
)
def plot_figure(VAE_model_name, measured_dataframe_json):
    if VAE_model_name in [None, "None"]:
        return ""

    se_profile = util.get_profile_df().loc[VAE_model_name]

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

    measured_df: pd.DataFrame = pd.read_json(measured_dataframe_json)

    if not "Without_Adapters" in measured_df:
        measured_df = measured_df[measured_df['Sequence'].apply(
            lambda seq: default_filterfunc(
                read = seq,
                fwd_adapter = se_profile["fwd_adapter"],
                rev_adapter = se_profile["rev_adapter"],
                target_length = se_profile["target_length"],
                tolerance = se_profile["target_length"],
            )
        )]
        measured_df['Without_Adapters'] = measured_df['Sequence'].apply(
            lambda seq: default_cutfunc(
                read = seq,
                fwd_adapter = se_profile["fwd_adapter"],
                rev_adapter = se_profile["rev_adapter"],
            )
        )
        coords_rep_x, coords_rep_y = np.array(embed_sequences(
            sequences=measured_df['Without_Adapters'].to_list(),
            model=VAE_model,
        )).T.tolist()
        measured_df['coord_x'] = coords_rep_x
        measured_df['coord_y'] = coords_rep_y

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

    fig.update_layout(
        legend = dict(
            yanchor="top", y=1, x=0,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        hoverlabel = dict(
            font_family="Courier New"
        ),
        clickmode='event+select'
    )

    plot.plot_SELEX(
        fig = fig,
        df_SELEX_data = unique_df,
        color = "silver",
    )

    measured_df = measured_df[["hue", "ID", "Sequence", "Without_Adapters", "coord_x", "coord_y"]]
    colors = ["cornflowerblue", "orange", "yellow", "purple", "red"]
    for idx, (hue_name, df) in enumerate(measured_df.groupby("hue")):
        plot.plot_data(
            fig = fig,
            series_name = hue_name,
            df_data = df,
            color = colors[idx % 5],
            columns_to_show = ["ID", "Without_Adapters"],
        hover_template = \
"""<b>%{customdata[1]}</b>
<b>Coord:</b> (%{x:.4f}, %{y:.4f})
<b>Seq:</b> %{customdata[2]}""".replace("\n", "<br>")
        )

    return dcc.Graph(figure=fig)

@callback(
    [
        Output("mu-submit-notifier-wrapper", "children"),
        Output("mu-submit-btn-wrapper", "children"),
    ],
    Input("mu-submit-btn", "n_clicks"),
    State("mu-measured-dataframe-json", "data"),
    State("mu-measured-file-name", "data"),
    prevent_initial_call=True,
)
def submit(n_clicks, measured_dataframe_json, filename):
    measured_df = pd.read_json(measured_dataframe_json)
    
    measured_df.to_csv(f"{util.get_data_dirname()}/measured_values/{Path(filename).stem}.csv")

    return (
        html.Div([
            html.H5("The table of measured values successfully submitted.")
        ], className="alert alert-dismissible alert-success"),
        ""
    )

storage: html.Div = html.Div([
    dcc.Store(id="mu-measured-dataframe-json"),
    dcc.Store(id="mu-measured-file-name"),
])

layout = dbc.Container([
    storage,
    dcc.Location(id='ou-url', refresh=True),
    html.H1("Upload Measured Data"),
    html.Hr(),
    html.Div(make_layout(), id="ou-page")
])

@callback(
    Output('ou-page', 'children'),
    Input('ou-url', 'href'),
)
def refresh_page(url):
    return [make_layout()]