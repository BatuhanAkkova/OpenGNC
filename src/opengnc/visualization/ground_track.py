
import numpy as np
import plotly.graph_objects as go


def plot_ground_track(
    latitudes: np.ndarray | list[float],
    longitudes: np.ndarray | list[float],
    times: np.ndarray | list[float] | None = None,
    title: str = "Satellite Ground Track"
) -> go.Figure:
    """
    2D Sub-Satellite Point (SSP) Visualization.

    Projects the spacecraft trajectory onto a 2D equirectangular map tracking
    latitude and longitude evolution over time.

    Parameters
    ----------
    latitudes : Union[np.ndarray, List[float]]
        Geodetic or geocentric latitudes (deg). Range: $[-90, 90]$.
    longitudes : Union[np.ndarray, List[float]]
        Geodetic or geocentric longitudes (deg). Range: $[-180, 180]$.
    times : Optional[Union[np.ndarray, List[float]]], optional
        Simulation timestamps (s) for hover metadata.
    title : str, optional
        Plot main heading.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 2D map plot.

    Raises
    ------
    ValueError
        If coordinate arrays have mismatched lengths.
    """
    lats = np.asarray(latitudes)
    lons = np.asarray(longitudes)

    if lats.shape != lons.shape:
        raise ValueError("Latitudes and longitudes must have the same length.")

    fig = go.Figure()

    # 1. Build Hover Metadata
    hover_text = []
    if times is not None:
        t_vals = np.asarray(times)
        for t, lat, lon in zip(t_vals, lats, lons):
            hover_text.append(f"T: {t:.1f}s<br>Lat: {lat:.2f}\u00b0<br>Lon: {lon:.2f}\u00b0")
    else:
        for lat, lon in zip(lats, lons):
            hover_text.append(f"Lat: {lat:.2f}\u00b0<br>Lon: {lon:.2f}\u00b0")

    # 2. Main Ground Track Trace
    fig.add_trace(
        go.Scattergeo(
            lat=lats, lon=lons,
            mode="lines+markers",
            line=dict(width=2, color="darkorange"),
            marker=dict(size=4, color="darkorange", opacity=0.8),
            hovertext=hover_text,
            hoverinfo="text",
            name="Ground Track",
        )
    )

    # 3. Markers
    fig.add_trace(
        go.Scattergeo(
            lat=[lats[0]], lon=[lons[0]],
            mode="markers",
            marker=dict(size=8, color="green", symbol="circle"),
            name="Start",
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lat=[lats[-1]], lon=[lons[-1]],
            mode="markers",
            marker=dict(size=8, color="red", symbol="square"),
            name="End",
        )
    )

    # 4. Map Layout configuration
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




