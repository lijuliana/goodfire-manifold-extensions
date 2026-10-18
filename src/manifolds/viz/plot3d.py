"""Goodfire-style 3-D manifold figures.

Look (after the figures in the Neural Geometry series): no axes, a light square floor plane
below the cloud with a soft grey shadow of the points, points coloured by the ordering variable,
monospace captions. Rendered with Plotly so the site stays static and interactive.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

FONT = "IBM Plex Mono, ui-monospace, Menlo, Consolas, monospace"
TEXT = "#1D272A"
FLOOR = "#EDEDED"
SHADOW = "#555555"
CONTINUOUS_SCALE = "Viridis"
CYCLIC_SCALE = "Phase"
DIVERGING_SCALE = [[0.0, "#C0392B"], [0.5, "#D9D9D9"], [1.0, "#2E5BFF"]]   # red -> grey -> blue


def colorscale_for(mode: str) -> Any:
    return {"continuous": CONTINUOUS_SCALE, "cyclic": CYCLIC_SCALE,
            "diverging": DIVERGING_SCALE}[mode]
CATEGORICAL = [
    "#2E5BFF", "#FF6B2E", "#2EB872", "#D62E8C", "#8E2EFF",
    "#FFB02E", "#2EC4E8", "#A3A3A3", "#6B4226", "#1D272A",
]


def subsample(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    """Deterministic index subset of size min(n, max_points)."""
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _floor_traces(coords: np.ndarray, point_size: float) -> list[go.BaseTraceType]:
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    span = float(np.ptp(coords, axis=0).max()) or 1.0
    floor_z = float(z.min() - 0.30 * span)
    cx, cy = float(x.mean()), float(y.mean())
    r = float(max(np.ptp(x), np.ptp(y)) / 2 * 1.15) or 1.0
    xs = [cx - r, cx + r, cx + r, cx - r]
    ys = [cy - r, cy - r, cy + r, cy + r]
    floor = go.Mesh3d(
        x=xs, y=ys, z=[floor_z] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
        color=FLOOR, opacity=0.85, flatshading=True, hoverinfo="skip",
        lighting={"ambient": 1.0, "diffuse": 0.0, "specular": 0.0},
        showscale=False, name="floor",
    )
    shadow = go.Scatter3d(
        x=x, y=y, z=np.full_like(z, floor_z + 1e-3 * span), mode="markers",
        marker={"size": point_size * 1.9, "color": SHADOW, "opacity": 0.07},
        hoverinfo="skip", showlegend=False, name="shadow",
    )
    return [floor, shadow]


def _colorbar(title: str) -> dict[str, Any]:
    return {
        "orientation": "h", "thickness": 7, "len": 0.42, "x": 0.5, "y": -0.04,
        "xanchor": "center", "yanchor": "top",
        "title": {"text": title, "side": "top",
                  "font": {"family": FONT, "size": 11, "color": TEXT}},
        "tickfont": {"family": FONT, "size": 10, "color": TEXT}, "outlinewidth": 0,
    }


def build_figure(
    coords: np.ndarray,
    color_values: Sequence[Any],
    *,
    color_mode: str,
    hover_text: Sequence[str],
    color_title: str = "",
    category_order: Sequence[Any] | None = None,
    point_size: float = 4.5,
    camera_eye: tuple[float, float, float] = (1.55, 1.25, 0.85),
) -> go.Figure:
    """Build the figure.

    color_mode: "continuous" (Viridis), "cyclic" (Phase), "diverging" (red-grey-blue),
    "categorical" (one trace per class with a legend) or "hex" (each point drawn in its own hex
    colour, e.g. the colors manifold).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be [N, 3]")
    if len(color_values) != len(coords) or len(hover_text) != len(coords):
        raise ValueError("color_values and hover_text must match coords length")
    traces: list[go.BaseTraceType] = _floor_traces(coords, point_size)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    base_marker = {"size": point_size, "opacity": 0.95, "line": {"width": 0}}
    showlegend = False

    if color_mode in ("continuous", "cyclic", "diverging"):
        vals = np.asarray(color_values, dtype=float)
        marker = base_marker | {
            "color": vals,
            "colorscale": colorscale_for(color_mode),
            "showscale": True,
            "colorbar": _colorbar(color_title),
        }
        traces.append(go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=marker,
                                   text=list(hover_text), hoverinfo="text", name="points"))
    elif color_mode == "hex":
        marker = base_marker | {"color": [str(c) for c in color_values]}
        traces.append(go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=marker,
                                   text=list(hover_text), hoverinfo="text", name="points"))
    elif color_mode == "categorical":
        cats = np.asarray(color_values)
        order = list(category_order) if category_order is not None else list(dict.fromkeys(cats))
        showlegend = True
        for i, c in enumerate(order):
            m = cats == c
            if not m.any():
                continue
            marker = base_marker | {"color": CATEGORICAL[i % len(CATEGORICAL)]}
            traces.append(go.Scatter3d(
                x=x[m], y=y[m], z=z[m], mode="markers", marker=marker, name=str(c),
                text=[hover_text[int(j)] for j in np.flatnonzero(m)], hoverinfo="text",
            ))
    else:
        raise ValueError(f"unknown color_mode {color_mode!r}")

    axis = {"visible": False, "showbackground": False, "showgrid": False, "zeroline": False,
            "showticklabels": False, "title": ""}
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene={"xaxis": axis, "yaxis": axis, "zaxis": axis, "aspectmode": "data",
               "camera": {"eye": {"x": camera_eye[0], "y": camera_eye[1], "z": camera_eye[2]},
                          "up": {"x": 0, "y": 0, "z": 1}},
               "bgcolor": "rgba(0,0,0,0)"},
        paper_bgcolor="white", plot_bgcolor="white",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        font={"family": FONT, "color": TEXT, "size": 12},
        showlegend=showlegend,
        legend={"font": {"family": FONT, "size": 11}, "orientation": "h", "x": 0.5, "y": -0.02,
                "xanchor": "center", "yanchor": "top", "itemsizing": "constant"},
        hoverlabel={"font": {"family": FONT, "size": 11}, "bgcolor": "white"},
    )
    return fig


def frames_figure(
    steps: list[tuple[str, np.ndarray]],
    color_values: Sequence[Any],
    *,
    color_mode: str,
    hover_text: Sequence[str],
    active: int = 0,
    prefix: str = "",
    hover_by_step: dict[str, Sequence[str]] | None = None,
    point_size: float = 3.8,
    camera_eye: tuple[float, float, float] = (1.55, 1.25, 0.85),
) -> go.Figure:
    """One 3-D scatter per step with a slider (used for ablation strength, neuron ablations,
    training checkpoints). Same look as `build_figure`; the floor is rebuilt per frame."""
    hover_by_step = hover_by_step or {}
    base_marker: dict[str, Any] = {"size": point_size, "opacity": 0.95, "line": {"width": 0}}
    if color_mode == "hex":
        base_marker["color"] = [str(c) for c in color_values]
    elif color_mode == "categorical":
        cats = list(dict.fromkeys(color_values))
        base_marker["color"] = [CATEGORICAL[cats.index(c) % len(CATEGORICAL)]
                                for c in color_values]
    else:
        base_marker["color"] = [float(c) for c in color_values]
        base_marker["colorscale"] = colorscale_for(color_mode)

    def traces(label: str, c: np.ndarray) -> list[go.BaseTraceType]:
        c = np.round(np.asarray(c, dtype=float), 3)
        txt = list(hover_by_step.get(label, hover_text))
        return _floor_traces(c, point_size) + [go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers", marker=base_marker, text=txt,
            hoverinfo="text", name=label)]

    fig = go.Figure(data=traces(*steps[active]))
    fig.frames = [go.Frame(data=traces(lab, c), name=lab) for lab, c in steps]
    slider_steps = [{"method": "animate", "label": lab,
                     "args": [[lab], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                      "transition": {"duration": 0}}]} for lab, _ in steps]
    axis = {"visible": False, "showbackground": False}
    fig.update_layout(
        scene={"xaxis": axis, "yaxis": axis, "zaxis": axis, "aspectmode": "data",
               "camera": {"eye": {"x": camera_eye[0], "y": camera_eye[1], "z": camera_eye[2]}}},
        sliders=[{"active": active, "steps": slider_steps, "x": 0.05, "len": 0.9, "y": 0.02,
                  "pad": {"t": 10}, "font": {"family": FONT, "size": 11},
                  "currentvalue": {"prefix": prefix, "font": {"family": FONT, "size": 12}}}],
        paper_bgcolor="white", margin={"l": 0, "r": 0, "t": 0, "b": 0},
        font={"family": FONT, "color": TEXT}, showlegend=False,
        hoverlabel={"font": {"family": FONT, "size": 11}, "bgcolor": "white"},
    )
    return fig


def line_layout(title_x: str, title_y: str, *, y_range: tuple[float, float] | None = None,
                legend_y: float = -0.3) -> dict[str, Any]:
    ax = {"showgrid": True, "gridcolor": "#eeeeee", "zeroline": False,
          "tickfont": {"family": FONT, "size": 11}}
    yax = ax | {"title": {"text": title_y}}
    if y_range is not None:
        yax["range"] = list(y_range)
    return {"paper_bgcolor": "white", "plot_bgcolor": "white",
            "font": {"family": FONT, "color": TEXT, "size": 12},
            "margin": {"l": 50, "r": 10, "t": 10, "b": 40},
            "xaxis": ax | {"title": {"text": title_x}}, "yaxis": yax,
            "legend": {"font": {"family": FONT, "size": 11}, "orientation": "h", "y": legend_y},
            "hoverlabel": {"font": {"family": FONT, "size": 11}}}


def figure_to_json(fig: go.Figure) -> str:
    out: str = fig.to_json()
    return out
