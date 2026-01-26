import plotly.graph_objects as go
import torch.nn as nn, torch, numpy as np
from plotly.subplots import make_subplots

from ..models.layers import PolynomialFilter


def _get_graph_filters(model: nn.Module) -> list[PolynomialFilter]:
    return [m for m in model.modules() if isinstance(m, PolynomialFilter)]



def plot_alphas(model: nn.Module, width: int = None, height: int = 600, raw: bool = False):
    "Plot alpha weights per head/layer with clean formatting and LaTeX support"
    layers = _get_graph_filters(model)
    attr = "alpha_raw" if raw else "alphas"
    alphas = torch.stack([getattr(l, attr).detach().cpu() for l in layers])  # [layers, alphas, heads]
    nl, na, nh = alphas.shape

    if alphas.min() >= 0:
        color_config = dict(colorscale="Blues", cmin=0, colorbar_title="Magnitude")
    else:
        color_config = dict(colorscale="RdBu_r", cmid=0, colorbar_title="Magnitude")

    width = width or max(500, nl * max(na * 60, 200))

    fig = make_subplots(1, nl, subplot_titles=[f"<b>Layer {i}</b>" for i in range(nl)], 
                        horizontal_spacing=0.05)

    for i, data in enumerate(alphas):
        fig.add_trace(go.Heatmap(
            z=data.T,  # shape: [heads, alphas]
            coloraxis="coloraxis",
            text=data.T, texttemplate="%{z:.2f}",
            textfont={"size": 10},
            hovertemplate="Head: %{y}<br>Alpha: %{x}<br>Val: %{z:.4f}<extra></extra>"
        ), 1, i+1)

    fig.update_layout(
        width=width, height=height, 
        margin=dict(t=150, b=50, l=100, r=50),
        coloraxis=color_config
    )

    for i in range(nl):
        fig.update_xaxes(
            tickvals=list(range(na)), 
            ticktext=[f"a{j}" for j in range(na)], 
            side="top", tickangle=0, col=i+1
        )
    for i in range(nl):
        fig.update_yaxes(
            autorange="reversed", tickvals=list(range(nh)), 
            ticktext=[f"h{j}" for j in range(nh)], 
            title_text="Head" if i == 0 else None, col=i+1
        )
    for ann in fig.layout.annotations: ann.update(y=1.12, font_size=14)

    fig.show(include_mathjax='cdn')
