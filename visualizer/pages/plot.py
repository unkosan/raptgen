from pandas import DataFrame
from raptgen.core.algorithms import *
from raptgen.core.preprocessing import *
from raptgen.core.train import *

from plotly.graph_objs._figure import Figure
from typing import Dict, Union
import plotly.colors
from PIL import ImageColor
import plotly.graph_objects as go
from _plotly_utils.basevalidators import ColorscaleValidator

def get_intermed_color(
    colorscale_name: str, 
    intermed: float
    ) -> Union[str, tuple]:
    """return color at the `intermed` value in the colorscale specified by `colorscale_name`

    Parameters
    ----------
    colorscale_name : str
        the name of continuous colorscale.
        available names: https://plotly.com/python/builtin-colorscales/#builtin-sequential-color-scales
    intermed : float
        needed to be an value between 0 and 1
    """
    
    cv = ColorscaleValidator("colorscale", "")
    colorscale = cv.validate_coerce(colorscale_name)

    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    low_cutoff, high_cutoff = 0, 1
    low_color, high_color = "#000000", "#FFFFFF"
    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor = low_color,
        highcolor = high_color,
        intermed = ((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype = "rgb",
    )

def plotEllipse(
    figure: Figure,
    mu: np.ndarray,
    coval: np.ndarray,
    center_text: str = "",
    template_text: str = "",
    ellipse_ID: str = "",
    color: Union[str, tuple] = "black"
    ) -> Figure:
    xcenter, ycenter = mu

    vals, vecs = np.linalg.eigh(coval)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.arctan2(*vecs[:, 0][::-1])
    nstd = 2
    w, h = nstd * np.sqrt(vals)

    t = np.linspace(0, 2*np.pi, 100)

    xs = w * np.cos(t)
    ys = h * np.sin(t)

    xp, yp = np.dot(
        [[ np.cos(theta), -np.sin(theta)],
         [ np.sin(theta),  np.cos(theta)]],
        [xs, ys],
    )

    x = xp + xcenter 
    y = yp + ycenter

    figure.add_scatter(
        x = x,
        y = y,
        mode = 'lines',
        name = '',
        text = ellipse_ID,
        hovertemplate = template_text,
        showlegend = False,
        line = dict(
            color = color,
        )
    )
    if center_text != "":
        figure.add_annotation(
            x = xcenter, y= ycenter,
            text = center_text,
            showarrow = False,
            hovertext = template_text,
            hoverlabel = dict(
                bgcolor=color
            )
        )

    return figure

def plot_SELEX(
    fig: Figure,
    df_SELEX_data: pd.DataFrame,
    color: str = "silver",
    ) -> Figure:

    assert hasattr(df_SELEX_data, "Without_Adapters")
    assert hasattr(df_SELEX_data, "Duplicates")
    assert hasattr(df_SELEX_data, "coord_x")
    assert hasattr(df_SELEX_data, "coord_y")

    df_SELEX_with_index: DataFrame = df_SELEX_data.reset_index()

    fig.add_trace(go.Scatter(
        mode = 'markers',
        x = df_SELEX_with_index["coord_x"],
        y = df_SELEX_with_index["coord_y"],
        marker = dict(
            size = df_SELEX_with_index["Duplicates"].apply(
                lambda val: max([2, np.sqrt(val)])
            ), 
            line = dict(
                color = color
            ),
            color = color
        ),
        text = ['SELEX data' for _ in range(len(df_SELEX_data))],
        name = 'SELEX data',
        showlegend = False,
        customdata = df_SELEX_with_index[["index", "Without_Adapters", "Duplicates"]],
        hovertemplate = \
"""<b>Coord:</b> (%{x:.4f}, %{y:.4f})
<b>Seq:</b> %{customdata[1]}
<b>Duplicates:</b> %{customdata[2]}""".replace("\n", "<br>")
        # customdata[0] to index
    ))

    return fig

def plot_data(
    fig: Figure,
    series_name: str,
    df_data: pd.DataFrame,
    columns_to_show: List[str],
    hover_template: str,
    color: Union[str, tuple] = "silver"
):

    assert hasattr(df_data, "coord_x")
    assert hasattr(df_data, "coord_y")

    df_data_with_index: DataFrame = df_data.reset_index()
    # df_data_with_index = df_data

    fig.add_trace(go.Scatter(
        mode = 'markers',
        x = df_data["coord_x"],
        y = df_data["coord_y"],
        marker = dict(
            size = 6,
            color = color
        ),
        text = [series_name for _ in range(len(df_data_with_index))],
        name = series_name,
        showlegend = True,
        customdata = df_data_with_index[["index"] + columns_to_show],
        hovertemplate = hover_template
        # customdata[0] to index
    ))

    return fig

def plot_GMM(
    fig: Figure,
    model: CNN_PHMM_VAE,
    GMM_params: Dict[str, np.ndarray],
    colorscale_name: str = "Viridis",
    ) -> Figure:

    assert "weights" in GMM_params.keys()
    assert "means" in GMM_params.keys()
    assert "covariances" in GMM_params.keys()
    assert "IDs" in GMM_params.keys()

    cmax = GMM_params["weights"].max()
    cmin = GMM_params["weights"].min()

    for id, weight, mean, coval in zip(
        GMM_params["IDs"],
        GMM_params["weights"],
        GMM_params["means"],
        GMM_params["covariances"],
    ):
        with torch.no_grad():
            model.eval()
            most_probable_seq = get_most_probable_seq(
                coords = [mean],
                model = model,
            )[0][0]

        plotEllipse(
            figure = fig,
            mu = mean,
            coval = coval,
            center_text = f"<b>{str(id)}</b>",
            ellipse_ID = f"MoG {id}",
            color = get_intermed_color(
                colorscale_name = colorscale_name,
                intermed = (weight - cmin) / (cmax - cmin),
            ),
            template_text = \
f'''<b>MoG No.{id}</b>
<b>Weight:</b>
{weight}
<b>Mean:</b>
{mean}
<b>Coval:</b>
{coval}
<b>Most Probable Seq:</b>
{most_probable_seq}'''.replace("\n", "<br>")
        )

    colorbar_trace = go.Scatter(
        x = [None],
        y = [None],
        mode = 'markers',
        marker = dict(
            colorscale=colorscale_name,
            showscale=True,
            cmin = cmin,
            cmax = cmax,
            colorbar = dict(
                thickness = 15,
                title = "GMM weight",
            )
        ),
        hoverinfo='none',
        showlegend = False
    )
    fig.add_trace(colorbar_trace)

    return fig


def plot_continuous_data(
    fig: Figure,
    series_name: str,
    df_data: pd.DataFrame,
    color_column: str,
    colorscale_name: str = "Viridis",
):

    assert hasattr(df_data, "coord_x")
    assert hasattr(df_data, "coord_y")

    fig.add_trace(go.Scatter(
        mode = 'markers',
        x = df_data["coord_x"],
        y = df_data["coord_y"],
        marker = dict(
            size = 6, 
            color = df_data[color_column],
            colorbar = dict(
                title = series_name,
            ),
            colorscale = colorscale_name
        ),
        text = [series_name for _ in range(len(df_data))],
        name = series_name,
        showlegend = True,
        customdata = df_data.index.values,
        # customdata as index
    ))

    return fig
