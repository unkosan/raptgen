from typing import List, Union
from dash import html, dcc
from dash.dash_table import DataTable
from pathlib import Path
import pandas as pd
import pickle
import torch
import io

DATA_DIR = "/raptgen-visualizer/visualizer/data"

# https://github.com/pytorch/pytorch/issues/16797
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def get_data_dirname():
    return DATA_DIR

def get_gmm_dataframe(VAE_model_name: str) -> pd.DataFrame:
    if Path(f"{DATA_DIR}/items/{VAE_model_name}/best_gmm_dataframe.pkl").exists():
        gmms_df = pd.read_pickle(f"{DATA_DIR}/items/{VAE_model_name}/best_gmm_dataframe.pkl")
    else:
        gmms_df = pd.DataFrame(data = None, columns = [
            "GMM_num_components",
            "GMM_seed",
            "GMM_optimal_model",
            "GMM_model_type",
        ])
        gmms_df.index.name = "name"
    
    return gmms_df

def get_profile_df() -> pd.DataFrame:
    if Path(f"{DATA_DIR}/profile_dataframe.pkl").exists():
        profile_df = pd.read_pickle(f"{DATA_DIR}/profile_dataframe.pkl")
    else:
        profile_df = pd.DataFrame(data = None, columns=[
            "published_time",
            "experiment",
            "round",
            "fwd_adapter",
            "rev_adapter",
            "target_length",
            "filtering_standard_length",
            "filtering_tolerance"
            "filtering_method",
            "minimum_count",
            "embedding_dim",
            "epochs",
            "beta_weight_epochs",
            "match_forcing_epochs",
            "match_cost",
            "early_stopping_epochs",
            "CUDA_num_workers",
            "CUDA_pin_memory",
            "pHMM_VAE_model_length",
            "pHMM_VAE_seed",
        ])
        profile_df.index.name = "name"

    return profile_df

def get_measured_value_names() -> List[str]:
    return [
        f.stem + f.suffix
        for f in Path(f"{DATA_DIR}/measured_values").iterdir()
    ]

def panda_table( id: str, columns: List[dict], data: List[dict]) -> DataTable:
    return DataTable(
        id = id,
        columns = columns,
        data = data,
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in ['Date', 'Region']
        ],
        style_data={
            'color': 'black',
            'backgroundColor': 'white'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(220, 220, 220)',
            }
        ],
        style_header={
            'backgroundColor': 'rgb(210, 210, 210)',
            'color': 'black',
            'fontWeight': 'bold'
        },
    )

def make_uploader(
    id: str,
    label: Union[str, html.Div] = html.Div(['Drag and Drop or ', html.A("Select Files")]),
) -> dcc.Upload:
    return dcc.Upload(
        id=id,
        children=label,
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px, 20px, 10px, 0px'
        },
        multiple=False
    )