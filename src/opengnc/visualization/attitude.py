import numpy as np
import plotly.graph_objects as go


def plot_attitude_sphere(
    vectors: np.ndarray | list[list[float]], title: str = "Attitude Sphere Visualization"
) -> go.Figure:
    r"""
    Directional Attitude Visualization.

    Projects vectors onto a unit sphere shell ($\|\mathbf{v}\| = 1$).

    Parameters
    ----------
    vectors : np.ndarray | list[list[float]]
        Sequence of 3D pointing vectors $(N, 3)$.
    title : str, optional
        Plot heading.

    Returns
    -------
    go.Figure
        Plotly 3D unit sphere object.
    """
    v_val = np.asarray(vectors)
    if v_val.ndim != 2 or v_val.shape[1] != 3:
        raise ValueError("vectors must be a 2D array of shape (N, 3)")

    # Normalize to unit shell
    norms = np.linalg.norm(v_val, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0  # Singularity guard
    v_norm = v_val / norms

    fig = go.Figure()

    # 1. Reference Unit Sphere
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale="Greys",
            opacity=0.15,
            showscale=False,
            name="Unit Sphere",
            hoverinfo="skip",
        )
    )

    # 2. Sequential Trace
    fig.add_trace(
        go.Scatter3d(
            x=v_norm[:, 0],
            y=v_norm[:, 1],
            z=v_norm[:, 2],
            mode="lines",
            line=dict(color="blue", width=4),
            name="Pointing Trace",
        )
    )

    # 3. Radial Pointer (Current State)
    curr = v_norm[-1]
    fig.add_trace(
        go.Scatter3d(
            x=[0, curr[0]],
            y=[0, curr[1]],
            z=[0, curr[2]],
            mode="lines+markers",
            line=dict(color="red", width=5),
            marker=dict(size=4),
            name="Current Vector",
        )
    )

    # 4. Landmarks
    fig.add_trace(
        go.Scatter3d(
            x=[v_norm[0, 0]],
            y=[v_norm[0, 1]],
            z=[v_norm[0, 2]],
            mode="markers",
            marker=dict(size=6, color="green"),
            name="Start",
        )
    )

    # 5. Scene constraints
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X", range=[-1.1, 1.1], gridcolor="lightgray"),
            yaxis=dict(title="Y", range=[-1.1, 1.1], gridcolor="lightgray"),
            zaxis=dict(title="Z", range=[-1.1, 1.1], gridcolor="lightgray"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(r=10, l=10, b=10, t=40),
    )

    return fig
