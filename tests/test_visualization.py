import pytest
import numpy as np
import plotly.graph_objects as go

from opengnc.visualization import (
    plot_orbit_3d,
    plot_attitude_sphere,
    plot_ground_track,
    plot_coverage_heatmap,
)


def test_plot_orbit_3d():
    r_eci = np.array([[7000000, 0, 0], [0, 7000000, 0], [0, 0, 7000000]])
    fig = plot_orbit_3d(r_eci)
    assert isinstance(fig, go.Figure)


def test_plot_orbit_3d_invalid():
    with pytest.raises(ValueError):
        plot_orbit_3d([1, 2, 3])  # invalid shape for list of 3-element lists


def test_plot_attitude_sphere():
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    fig = plot_attitude_sphere(vectors)
    assert isinstance(fig, go.Figure)


def test_plot_ground_track():
    lats = [0, 45, -45]
    lons = [0, 90, -90]
    fig = plot_ground_track(lats, lons)
    assert isinstance(fig, go.Figure)

    fig_times = plot_ground_track(lats, lons, times=[0, 10, 20])
    assert isinstance(fig_times, go.Figure)


def test_plot_ground_track_mismatch():
    with pytest.raises(ValueError):
        plot_ground_track([0, 45], [0])


def test_plot_coverage_heatmap():
    lats = [10, 20, 30]
    lons = [45, 90, 135]
    values = [1.0, 2.0, 3.0]
    fig = plot_coverage_heatmap(lats, lons, values)
    assert isinstance(fig, go.Figure)


def test_plot_attitude_sphere_invalid():
    with pytest.raises(ValueError):
        plot_attitude_sphere(np.array([1, 0, 0]))  # 1D array instead of 2D


def test_plot_coverage_heatmap_mismatch():
    with pytest.raises(ValueError):
        plot_coverage_heatmap([10, 20], [45, 90], [1.0])  # Shape mismatch
