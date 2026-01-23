import plotly.graph_objects as go
import torch.nn as nn, torch, numpy as np
from plotly.subplots import make_subplots

from ..models.layers import PolynomialFilter


def _get_graph_filters(model: nn.Module) -> list[PolynomialFilter]:
    return [m for m in model.modules() if isinstance(m, PolynomialFilter)]



def plot_alphas(model: nn.Module, width: int = None, height: int = 600):
    "Plot alpha weights per head/layer with clean formatting and LaTeX support"
    layers = _get_graph_filters(model)
    # [heads, layers, alphas]
    alphas = torch.stack([l.alphas.detach().cpu() for l in layers]).permute(2, 0, 1)
    nh, nl, na = alphas.shape
    
    # Ensure minimum 80px per alpha column to prevent text overlap
    width = width or max(800, nh * na * 80)

    fig = make_subplots(1, nh, subplot_titles=[f"<b>Head {i}</b>" for i in range(nh)], 
                        horizontal_spacing=0.05)

    for i, data in enumerate(alphas):
        fig.add_trace(go.Heatmap(
            z=data, coloraxis="coloraxis",
            text=data, texttemplate="%{z:.2f}", # Forces 2 decimal places in cells
            textfont={"size": 10},
            hovertemplate="Layer: %{y}<br>Alpha: %{x}<br>Val: %{z:.4f}<extra></extra>"
        ), 1, i+1)

    fig.update_layout(
        width=width, height=height, 
        margin=dict(t=150, b=50, l=100, r=50), # Large top margin for titles + labels
        coloraxis=dict(colorscale="RdBu_r", cmid=0, colorbar_title="Magnitude")
    )

    # side="top" moves labels up; tickangle=0 prevents the slanted overlap
    fig.update_xaxes(
        tickvals=list(range(na)), 
        ticktext=[f"a{j}" for j in range(na)], 
        side="top", tickangle=0
    )
    
    # autorange="reversed" puts Layer 0 at the top
    fig.update_yaxes(autorange="reversed", tickvals=list(range(nl)))
    fig.update_yaxes(title_text="Layer", col=1)

    # Move 'Head X' titles higher so they don't crash into alpha labels
    for ann in fig.layout.annotations: ann.update(y=1.12, font_size=14)

    fig.show(include_mathjax='cdn')
