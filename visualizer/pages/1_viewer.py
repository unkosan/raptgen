from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
from raptgen.core.algorithms import *
from raptgen.core.preprocessing import *
from raptgen.core.train import *
import pandas as pd
import pickle
from raptgen.core.algorithms import CNN_PHMM_VAE
from sklearn.mixture import GaussianMixture
import dash_bootstrap_components as dbc
import tempfile
import subprocess
import numpy as np

import re
import json
import base64

import matplotlib.pyplot as plt

import dash
from dash import html, dcc, Input, Output, State, callback
from dash.dash_table.Format import Format, Scheme

import plotly.graph_objects as go

from . import plot
from . import util

dash.register_page(
    __name__,
    path='/',
    title='RaptGen: Viewer',
    name='Viewer'
)

storage: html.Div = html.Div([
    dcc.Store(id='df-data-memory'),
    dcc.Store(id='df-eval-memory'),
    dcc.Store(id='GMM-params-memory'),
])

def make_sidebar():
    return html.Div([
        html.H4("Selected VAE Model"),
        html.P([
            dcc.Dropdown(
                id="VAE_model_name",
                options=[
                    {"label": entry, "value": entry}
                    for entry in util.get_profile_df().index
                ] + [
                    {"label": "None", "value": "None"}
                ],
                value="None",
            ),
        ]),
        html.H4("Selected GMM Model"),
        html.P([
            dcc.Dropdown(
                id="GMM_model_name",
                options=[
                    {"label": "None", "value": "None"}
                ],
                value="None",
            )
        ]),
        dbc.Switch(
            id="GMM_plot_switch",
            value=False,
            label="Plot GMM"
        ),
        html.Div(id="download"),
        html.H4("Selected experimental value"),
        html.P([
            dcc.Dropdown(
                id="eval_dataset_name",
                options=[
                    {"label": entry, "value": entry}
                    for entry in util.get_measured_value_names()
                ] + [
                    {"label": "None", "value": "None"}
                ],
                value="None"
            )
        ]),
        html.H4("Minimum Count to Show"),
        dcc.Slider(
            id="min_count",
            min=1, max=4, step=1, value=2,
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.H4("Input new sequence to encode"),
        dbc.Row([
            dbc.Col([
                html.P(dbc.Input(id="input_new_seq", placeholder="Enter sequence to encode"))
            ]),
            dbc.Col([
                html.P(dbc.Button("Encode", id="encode_new_seq", className="me-1"))
            ], width="auto")
        ]),
        html.P(
            util.make_uploader("upload-fasta", "Drag and Drop FASTA file"),
        ),

        html.H4("VAE Model Properties"),
        html.P([
            html.Label("VAE model not selected"),
        ], id="properties_table_div", style={"overflow": "scroll"}),
        html.H4("GMM Model Properties"),
        html.P([
            html.Label("GMM model not selected")
        ], id="GMM_properties_table_div", style={"overflow": "scroll"})
    ])

def make_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                make_sidebar()
            ], width=4),
            dbc.Col([
                # display
                dcc.Graph(id="figure"),
                # dcc.Tooltip(id="figure-tooltip"),
                html.Div(id="weblogo"),
                html.Hr(),
                html.Div("Data Not Selected", id="selected_table"),
            ])
        ])
    ])

layout = dbc.Container([
    storage,
    dcc.Location(id='url', refresh=True),
    html.H1("Viewer"),
    html.Hr(),
    html.Div(make_layout(), id="page"),
])

@callback(
    Output('page', 'children'),
    Input('url', 'href'),
)
def refresh_page(url):
    return [make_layout()]


@callback(
    [ Output("download", "children") ],
    [ Input("GMM_plot_switch", "value") ],
    [
        State("VAE_model_name", "value"),
        State("GMM_model_name", "value"),
    ],
    prevent_initial_call = True
)
def show_download(switch: bool, VAE_model_name: str, GMM_model_name: str):
    if switch == True:
        df_GMM = util.get_gmm_dataframe(VAE_model_name)
        se_GMM = df_GMM.loc[GMM_model_name]
        children = html.Div([ 
            dcc.Download(id = "download_handler"),
            dbc.Row([
                dbc.Col(
                    html.P([
                        dbc.Label("Download cluster", html_for="download_dropdown"),
                        dcc.Dropdown(
                            options = ["all"] + list(map(str, range(se_GMM["GMM_num_components"]))),
                            value = "all",
                            id = "download_dropdown",
                        ),
                    ])
                ),
                dbc.Col(
                    html.P([
                        dbc.Button("Download", "download_btn", className="me-1"),
                    ])
                , width="auto", align="end")
            ]),
        ])
        return [children]
    else:
        return [""]

@callback(
    [ Output("download_handler", "data") ],
    [ Input("download_btn", "n_clicks") ],
    [
        State("download_dropdown", "value"),
        State("VAE_model_name", "value"),
        State("GMM_model_name", "value"),
    ],
    prevent_initial_call = True
)
def download(
    n_clicks: int,
    dropdown_value: str,
    VAE_name: str,
    GMM_name: str,
):
    df_filtered: pd.DataFrame = pd.read_pickle( f"{util.get_data_dirname()}/items/{VAE_name}/unique_seq_dataframe.pkl" )
    df_GMM = util.get_gmm_dataframe(VAE_name)
    # df_GMM: pd.DataFrame = pd.read_pickle(
    #     f"{util.get_data_dirname()}/items/{VAE_name}/best_gmm_dataframe.pkl"
    # )
    GMM_model: GaussianMixture = df_GMM.loc[GMM_name]["GMM_optimal_model"]

    data = np.array((df_filtered["coord_x"], df_filtered["coord_y"])).T
    df_copy = df_filtered.copy()
    df_copy["class"] = GMM_model.predict(data)
    if dropdown_value == "all":
        ret_df = df_copy
        name = f"{GMM_name}_all.csv"
    else:
        cluster_num = int(dropdown_value)
        ret_df = df_copy[df_copy["class"] == cluster_num]
        name = f"{GMM_name}_cluster_{cluster_num}.csv"

    return [dcc.send_data_frame(ret_df.to_csv, name)]


@callback(
    [
        Output("GMM_plot_switch", "disabled"),
        Output("GMM_plot_switch", "value")
    ],
    Input("GMM_model_name", "value")
)
def activate_plot_GMM_toggle(GMM_model_name):
    if GMM_model_name in ["None", None]:
        return True, False
    else:
        return False, True

@callback(
    Output("properties_table_div", "children"),
    Input("VAE_model_name", "value"),
)
def initialize_VAE_table(VAE_model_name):
    if VAE_model_name in ["None", None]:
        return "VAE model not selected"

    se_profile = util.get_profile_df().loc[VAE_model_name]
    VAE_table_div_children = html.P(
        util.panda_table(
            id="VAE_properies_table",
            columns=[
                {"name": "Item", "id": "Prop_Item"},
                {"name": "Value", "id": "Prop_Value"}],
            data=[
                {"Prop_Item": key, "Prop_Value": value}
                for key, value in se_profile.to_dict().items()
            ],
        )
    )

    return VAE_table_div_children

@callback(
    [
        Output("GMM_model_name", "options"),
        Output("GMM_model_name", "value"),
    ],
    Input("VAE_model_name", "value"),
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
    Output("weblogo", "children"),
    [
        Input("figure", "clickData"),
        Input("VAE_model_name", "value"),
    ],
)
def print_weblogo( hovered_data, VAE_model_name ):
    if hovered_data == None:
        return "Click on the point to show weblogo and secondary structure"

    se_profile = util.get_profile_df().loc[VAE_model_name]
    fwd_a = se_profile["fwd_adapter"]
    rev_a = se_profile["rev_adapter"]
    
    with Path(f"{util.get_data_dirname()}/items/{VAE_model_name}/VAE_model.pkl").open("rb") as f:
        state_dict = util.CPU_Unpickler(f).load()
    model = CNN_PHMM_VAE(
        motif_len=se_profile["pHMM_VAE_model_length"],
        embed_size=se_profile["embedding_dim"]
    )
    model.load_state_dict(state_dict)
    model.eval()

    x_val = hovered_data["points"][0]["x"]
    y_val = hovered_data["points"][0]["y"]
    coord = np.array([x_val, y_val])

    # weblogo

    with tempfile.NamedTemporaryFile("w+b", suffix=".png") as fp:
        fig, ax = plt.subplots(1,1,figsize=(10,3),dpi=120)
        draw_logo(
            ax = ax,
            coord = np.array(coord),
            model = model,
        )
        fig.savefig(fp.name, format="png")
        read = fp.read()
        encoded_weblogo = base64.b64encode(read)

    # secondary structure
    seq = hovered_data['points'][0]['customdata'][1]
    # seq = get_most_probable_seq(
    #     [coord],
    #     model = model,
    # )[0][0].replace("N", "").replace("_", "")

    with tempfile.NamedTemporaryFile("w+", suffix=".fasta") as tempf_fasta, \
         tempfile.NamedTemporaryFile("w+", suffix=".ps") as tempf_ps, \
         tempfile.NamedTemporaryFile("w+b", suffix=".png") as tempf_png:
        tempf_fasta.write(f">\n{fwd_a}{seq}{rev_a}")
        tempf_fasta.flush()
        subprocess.run(["centroid_fold", tempf_fasta.name, "--postscript", tempf_ps.name], stdout=subprocess.PIPE)
        subprocess.run(["gs", "-o", tempf_png.name, "-sDEVICE=pngalpha", tempf_ps.name], stdout=subprocess.PIPE)
        encoded_ss = base64.b64encode(tempf_png.read())

    return [
        dbc.Row([
            dbc.Col([
                html.H4("Weblogo"),
                html.Div([
                    html.Img(src="data:image/png;charset=utf-8;base64," + encoded_weblogo.decode(),  style={'width':'100%'}),
                ])
            ]),
            dbc.Col([
                html.H4("Secondary Structure"),
                html.Img(src="data:image/svg;charset=utf-8;base64," + encoded_ss.decode(),  style={'width':'100%'}),
            ])
        ])
    ]

@callback(
    Output("selected_table", "children"),
    [
        Input("figure", "selectedData"),
        Input("df-data-memory", "data"),
        Input("df-eval-memory", "data"),
    ]
)
def print_selected_datatable(
    selected_data, 
    df_data_json, 
    df_eval_json,
):
    if selected_data == None:
        return "Data Not Selected"

    df_data = pd.read_json(df_data_json)
    df_eval = pd.read_json(df_eval_json)
    SELEX_IDs = list()
    eval_IDs = defaultdict(list)


    for point_dict in selected_data["points"]:
        if point_dict["text"] == "SELEX data":
            SELEX_IDs.append(point_dict["customdata"][0])
        else:
            eval_IDs[point_dict["text"]].append(point_dict["customdata"][0])
    
    df_data = df_data.loc[SELEX_IDs]
    normal_table = util.panda_table(
        id = 'normal_table',
        columns = [
            {"id": "Without_Adapters", "name": "Sequence"},
            {"id": "Duplicates", "name": "Duplicates"},
            {"id": "coord_x", "name": "coord_x",
                "type": "numeric",
                "format": Format(
                    precision=4,
                    scheme=Scheme.fixed
                )
            },
            {"id": "coord_y", "name": "coord_y",
                "type": "numeric",
                "format": Format(
                    precision=4,
                    scheme=Scheme.fixed
                )
            },
        ],
        data = df_data[["Without_Adapters", "Duplicates", "coord_x", "coord_y"]].to_dict('records'),
    )

    tables = [
        html.H4("Selected SELEX Data"),
        normal_table,
        html.Hr(),
    ]

    if len(df_eval) != 0:
        for hue_name, locs in eval_IDs.items():
            tables.append(html.H4(f"Selected {hue_name}"))
            tables.append(util.panda_table(
                id = f"{hue_name}_table",
                columns = [
                    {"id": "ID", "name": "ID"},
                    {"id": "Without_Adapters", "name": "Sequence"},
                    {"id": "coord_x", "name": "coord_x",
                        "type": "numeric",
                        "format": Format(
                            precision=4,
                            scheme=Scheme.fixed
                        )
                    },
                    {"id": "coord_y", "name": "coord_y",
                        "type": "numeric",
                        "format": Format(
                            precision=4,
                            scheme=Scheme.fixed
                        )
                    },
                ],
                data = df_eval.loc[locs][["ID", "Without_Adapters", "coord_x", "coord_y"]].to_dict("records"),
            ))
            tables.append(html.Hr())

    return tables
    
@callback(
    Output("GMM_properties_table_div", "children"),
    [
        Input("VAE_model_name", "value"),
        Input("GMM_model_name", "value")
    ],
)
def print_GMM_properties_table(VAE_model_name, GMM_model_name):
    if GMM_model_name in ["None", None]:
        GMM_table = html.Label("GMM model not selected")
    else:
        df_GMM: pd.DataFrame = util.get_gmm_dataframe(VAE_model_name) 
        se_GMM = df_GMM.loc[GMM_model_name]
        GMM_table = util.panda_table(
            id="GMM_properies_table",
            columns=[
                {"name": "Item", "id": "GMM_Prop_Item"},
                {"name": "Value", "id": "GMM_Prop_Value"}],
            data=[
                {"GMM_Prop_Item": key, "GMM_Prop_Value": str(value)}
                for key, value in se_GMM.drop("GMM_optimal_model").to_dict().items()
            ],
        )
    return html.P(GMM_table)

@callback(
    [
        Output("figure", "figure"),
        Output("df-data-memory", "data"),
        Output("df-eval-memory", "data"),
        Output("GMM-params-memory", "data")
    ], [
        Input("VAE_model_name", "value"),
        Input("GMM_model_name", "value"),
        Input("eval_dataset_name", "value"),
        Input("min_count", "value"),
        Input("GMM_plot_switch", "value"),
        Input("encode_new_seq", "n_clicks"),
        Input("upload-fasta", "contents"),
    ],
    State("input_new_seq", "value"),
    prevent_initial_call=True,
)
def plot_figure( VAE_model_name, GMM_model_name, eval_name, min_count, GMM_plot_switch, n_clicks, fasta_content, input_new_seq ):
    
    fig = go.Figure(
        layout = dict(
            height = 800,
            title = VAE_model_name,
            template = "ggplot2",
            yaxis = dict(
                scaleanchor = "x"
            )
        )
    )

    if VAE_model_name in ["None", None]:
        return (
            fig,
            None,
            None,
            None,
        )

    se_profile = util.get_profile_df().loc[VAE_model_name]
    with Path(f"{util.get_data_dirname()}/items/{VAE_model_name}/VAE_model.pkl").open("rb") as f:
        state_dict = util.CPU_Unpickler(f).load()
    model = CNN_PHMM_VAE(
        motif_len=se_profile["pHMM_VAE_model_length"],
        embed_size=se_profile["embedding_dim"]
    )
    model.load_state_dict(state_dict)
            
    df_filtered = pd.read_pickle( f"{util.get_data_dirname()}/items/{VAE_model_name}/unique_seq_dataframe.pkl" )
    df_filtered = df_filtered[
        df_filtered["Duplicates"] >= min_count
    ]

    fig.update_xaxes(range=[-3.5, 3.5])
    fig.update_yaxes(range=[-3.5, 3.5])

    plot.plot_SELEX(
        fig = fig,
        df_SELEX_data = df_filtered,
        color = "silver"
    )

    GMM_params_alt: Union[Dict[str, list], None] = None

    if GMM_plot_switch == True:
        GMM_df_path = f"{util.get_data_dirname()}/items/{VAE_model_name}/best_gmm_dataframe.pkl"
        df_GMM: pd.DataFrame = pd.read_pickle(GMM_df_path)
        se_GMM = df_GMM.loc[GMM_model_name]
        best_gmm = se_GMM["GMM_optimal_model"]

        GMM_params: Dict[str, np.ndarray] = {
            "IDs": np.array(range(len(best_gmm.means_))),
            "weights": best_gmm.weights_,
            "means": best_gmm.means_,
            "covariances": best_gmm.covariances_,
        }

        GMM_params_alt = {
            key: value.tolist() for key, value in GMM_params.items()
        }

        plot.plot_GMM(
            fig = fig,
            model = model,
            GMM_params = GMM_params,
            colorscale_name = "Viridis",

        )
        
    #########################################################################
    if eval_name not in ["None", None]:
        df_eval = pd.read_csv(f"{util.get_data_dirname()}/measured_values/{eval_name}")
        assert "Sequence" in df_eval.columns
        assert "ID" in df_eval.columns
        assert "hue" in df_eval.columns

        if not "Without_Adapters" in df_eval.columns:
            df_eval = df_eval[df_eval['Sequence'].apply(
                lambda seq: default_filterfunc(
                    read = seq,
                    fwd_adapter = se_profile["fwd_adapter"],
                    rev_adapter = se_profile["rev_adapter"],
                    target_length = se_profile["target_length"],
                    tolerance = se_profile["target_length"],
                )
            )]
            df_eval['Without_Adapters'] = df_eval['Sequence'].apply(
                lambda seq: default_cutfunc(
                    read = seq,
                    fwd_adapter = se_profile["fwd_adapter"],
                    rev_adapter = se_profile["rev_adapter"],
                )
            )
            coords_rep_x, coords_rep_y = np.array(embed_sequences(
                sequences=df_eval['Without_Adapters'].to_list(),
                model=model,
            )).T.tolist()
            df_eval['coord_x'] = coords_rep_x
            df_eval['coord_y'] = coords_rep_y
        df_eval = df_eval[["hue", "ID", "Sequence", "Without_Adapters", "coord_x", "coord_y"]]
        colors = ["cornflowerblue", "orange", "yellow", "purple", "red"]
        for idx, (hue_name, df) in enumerate(df_eval.groupby("hue")):
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
    else:
        df_eval = pd.DataFrame()
        
    #########################################################################


    if fasta_content != None:
        content_type, content_string = fasta_content.split(',')
        # try:
        with tempfile.NamedTemporaryFile("w+", suffix=".fasta") as fp:
            fp.write(base64.b64decode(content_string).decode("utf8"))
            fp.flush()
            df_new_fasta = read_SELEX_data(
                filepath=fp.name,
                filetype="fasta",
            )
        coords_new_fasta_x, coords_new_fasta_y = np.array(embed_sequences(
            sequences=df_new_fasta["Sequence"].to_list(),
            model = model,
        )).T.tolist()
        df_new_fasta["coord_x"] = coords_new_fasta_x
        df_new_fasta["coord_y"] = coords_new_fasta_y
        plot.plot_data(
            fig = fig,
            series_name = "Input FASTA",
            df_data = df_new_fasta,
            color = "lightgreen",
            columns_to_show = ["ID", "Sequence"],
            hover_template = \
"""<b>%{customdata[1]}</b>
<b>Coord:</b> (%{x:.4f}, %{y:.4f})
<b>Seq:</b> %{customdata[2]}""".replace("\n", "<br>")
        )

        # except:
            # print("something wrong happened for input fasta file")

    if type(input_new_seq) == str \
        and re.compile("[ACGTU]+").fullmatch(input_new_seq) != None:
        input_new_seq.replace("T", "U")
        coords_new_seq_x, coords_new_seq_y = np.array(embed_sequences(
            sequences=[ input_new_seq ],
            model=model,
        )).T.tolist()
        df_new_seq = pd.DataFrame({
            "coord_x": coords_new_seq_x,
            "coord_y": coords_new_seq_y,
            "ID": ["Input sequence"],
            "Without_Adapters": [input_new_seq]
        })
        plot.plot_data(
            fig = fig,
            series_name = "Input Sequence",
            df_data = df_new_seq,
            color = "green",
            columns_to_show = ["ID", "Without_Adapters"],
            hover_template = \
"""<b>%{customdata[1]}</b>
<b>Coord:</b> (%{x:.4f}, %{y:.4f})
<b>Seq:</b> %{customdata[2]}""".replace("\n", "<br>")
        )


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

    return (
        fig,
        df_filtered.to_json(), 
        df_eval.to_json(), 
        json.dumps(GMM_params_alt),
    )
