import numpy as np
import plotly.graph_objects as go


def plot_coverage_heatmap(
    latitudes: np.ndarray | list[float],
    longitudes: np.ndarray | list[float],
    values: np.ndarray | list[float],
    title: str = "Coverage Heat Map",
) -> go.Figure:
    """
    Plots a density/heatmap on an Earth map to analyze access/coverage.

    Parameters
    ----------
        latitudes (numpy.ndarray or list): Flattened list of latitudes [deg].
        longitudes (numpy.ndarray or list): Flattened list of longitudes [deg].
        values (numpy.ndarray or list): Corresponding continuous coverage values (e.g., access time, frequency).
        title (str): Title for the plot.

    Returns
    -------
        plotly.graph_objects.Figure: The plotly figure object.
    """
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    values = np.array(values)

    if latitudes.shape != longitudes.shape or latitudes.shape != values.shape:
        raise ValueError("Latitudes, longitudes, and values must have the same length/shape.")

    fig = go.Figure()

    # 1. Add Scattergeo with Color scale for coverage values
    fig.add_trace(
        go.Scattergeo(
            lat=latitudes,
            lon=longitudes,
            mode="markers",
            marker=dict(
                size=7,
                color=values,
                colorscale="Viridis",  # Good for continuous values
                colorbar=dict(title="Value"),
                opacity=0.4,  # Make translucent to see map below
                showscale=True,
            ),
            name="Coverage Value",
            hoverinfo="skip" if True else "text",  # Can add hover details if needed
        )
    )

    # 2. Update Layout
    fig.update_layout(
        title=title,
        geo=dict(
            showland=True,
            showcoastlines=True,
            projection_type="equirectangular",
            coastlinecolor="gray",
            landcolor="white",
            lakecolor="lightblue",
            rivercolor="lightblue",
            showocean=True,
            oceancolor="aliceblue",
            lataxis=dict(range=[-90, 90], showgrid=True, gridcolor="lightgray"),
            lonaxis=dict(range=[-180, 180], showgrid=True, gridcolor="lightgray"),
        ),
        margin=dict(r=0, l=0, b=0, t=40),
    )

    return fig
