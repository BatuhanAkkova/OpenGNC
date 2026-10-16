import numpy as np
import plotly.graph_objects as go


def plot_orbit_3d(
    r_eci: np.ndarray | list[list[float]],
    r_earth: bool = True,
    title: str = "3D Orbit Visualization",
) -> go.Figure:
    """
    Standard 3D Orbit Visualization.

    Renders flight path relative to a spherical Earth ($R_e = 6378.137$ km).

    Parameters
    ----------
    r_eci : np.ndarray | list[list[float]]
        Sequence of ECI positions $(N, 3)$ (m).
    r_earth : bool, optional
        Render Earth sphere. Default True.
    title : str, optional
        Plot heading.

    Returns
    -------
    go.Figure
        Plotly 3D scatter object.
    """
    r_val = np.asarray(r_eci)

    if r_val.ndim != 2 or r_val.shape[1] != 3:
        raise ValueError("r_eci must be a 2D array of shape (N, 3)")

    fig = go.Figure()

    # 1. Earth Sphere
    if r_earth:
        R_Earth = 6378137.0
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        x = R_Earth * np.outer(np.cos(u), np.sin(v))
        y = R_Earth * np.outer(np.sin(u), np.sin(v))
        z = R_Earth * np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale="Blues",
                opacity=0.3,
                showscale=False,
                name="Earth",
                hoverinfo="skip",
            )
        )

    # 2. Flight Path
    fig.add_trace(
        go.Scatter3d(
            x=r_val[:, 0],
            y=r_val[:, 1],
            z=r_val[:, 2],
            mode="lines",
            line=dict(color="darkred", width=4),
            name="Orbit Path",
        )
    )

    # 3. Markers
    fig.add_trace(
        go.Scatter3d(
            x=[r_val[0, 0]],
            y=[r_val[0, 1]],
            z=[r_val[0, 2]],
            mode="markers",
            marker=dict(size=6, color="green", symbol="circle"),
            name="Start",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[r_val[-1, 0]],
            y=[r_val[-1, 1]],
            z=[r_val[-1, 2]],
            mode="markers",
            marker=dict(size=6, color="orange", symbol="circle"),
            name="Current Position",
        )
    )

    # 4. Global Scene Format
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X [m]", gridcolor="gray", showbackground=False),
            yaxis=dict(title="Y [m]", gridcolor="gray", showbackground=False),
            zaxis=dict(title="Z [m]", gridcolor="gray", showbackground=False),
            aspectmode="data",
        ),
        margin=dict(r=10, l=10, b=10, t=40),
    )

    return fig
