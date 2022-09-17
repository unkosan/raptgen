from pathlib import Path
import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

from pages.util import DATA_DIR

if not DATA_DIR.exists():
    DATA_DIR.mkdir()
    (DATA_DIR / "items").mkdir()
    (DATA_DIR / "measured_values").mkdir()

def make_navbar(pages):
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink(page['name'], href=page["path"], active="exact"))
            for page in pages
        ],
        brand="RaptGen Visualizer",
        color="primary",
        dark=True
    )

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.COSMO],
    suppress_callback_exceptions=True,
    use_pages=True,
)

app.layout = html.Div([
    make_navbar(dash.page_registry.values()),
    html.Div("", style={"height": "20px"}),
    dbc.Container(dash.page_container),
])

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port="8050")