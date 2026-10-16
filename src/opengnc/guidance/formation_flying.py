"""
Formation Flying Control routines (Virtual Structure, Leader-Follower,
Fuel-Balanced, and Distributed Consensus).
"""

import numpy as np


def virtual_structure_control(
    state_actual: np.ndarray, state_desired: np.ndarray, gains: np.ndarray
) -> np.ndarray:
    """
    Compute control inputs to track a virtual structure formation.

    Parameters
    ----------
    state_actual : np.ndarray
        Current states of the spacecraft ensemble (N x state_dim).
    state_desired : np.ndarray
        Desired states derived from the virtual structure (N x state_dim).
    gains : np.ndarray
        Feedback gains (Proportional/Derivative or state-feedback).

    Returns
    -------
    np.ndarray
        Computed control actions for each spacecraft (N x control_dim).
    """
    # Simple proportional feedback to desired formation state
    return np.asarray(-gains * (np.asarray(state_actual) - np.asarray(state_desired)))


def leader_follower_control(
    leader_state: np.ndarray,
    follower_state: np.ndarray,
    desired_relative_state: np.ndarray,
    gains: np.ndarray,
) -> np.ndarray:
    """
    Compute control command for a follower to maintain offset from a leader.

    Parameters
    ----------
    leader_state : np.ndarray
        State of the leader spacecraft.
    follower_state : np.ndarray
        State of the follower spacecraft.
    desired_relative_state : np.ndarray
        Target offset of follower from leader (follower_target - leader).
    gains : np.ndarray
        Control feedback gains.

    Returns
    -------
    np.ndarray
        Control command for the follower.
    """
    target_state = leader_state + desired_relative_state
    return np.asarray(-gains * (follower_state - target_state))


def fuel_balanced_formation_keeping(
    states: np.ndarray, fuel_levels: np.ndarray, base_weights: np.ndarray
) -> np.ndarray:
    """
    Distribute control effort among formation to balance fuel consumption.

    Spacecraft with higher fuel levels are assigned a larger proportion of
    the control effort to equalize fuel mass across the ensemble over time.

    Parameters
    ----------
    states : np.ndarray
        (N x M) array of states for N spacecraft.
    fuel_levels : np.ndarray
        (N,) array of current fuel mass or percentages.
    base_weights : np.ndarray
        (N, M) base control actions or priority weights.

    Returns
    -------
    np.ndarray
        Fuel-balanced control actions.
    """
    fuel_norm = np.asarray(fuel_levels)
    total_fuel = np.sum(fuel_norm)
    if total_fuel < 1e-9:
        return np.asarray(np.zeros_like(base_weights))

    fuel_ratios = fuel_norm / total_fuel
    # Scale control weights by available fuel fraction
    return np.asarray(np.asarray(base_weights) * fuel_ratios[:, np.newaxis])


def distributed_consensus_control(
    states: np.ndarray, laplacian_matrix: np.ndarray, gains: np.ndarray
) -> np.ndarray:
    r"""
    Compute consensus control for a distributed multi-agent system.

    Uses the graph Laplacian matrix to drive agents towards a common state
    through local interactions only.
    $u_i = -K \sum_{j} a_{ij} (x_i - x_j)$

    Parameters
    ----------
    states : np.ndarray
        (N, M) array of states for N agents.
    laplacian_matrix : np.ndarray
        (N, N) graph Laplacian matrix ($L = D - A$).
    gains : np.ndarray
        Gains applied to the consensus error protocol.

    Returns
    -------
    np.ndarray
        (N, M) array of control inputs.
    """
    # The term (L @ X) computes the sum of relative state errors in the network
    consensus_error = np.asarray(laplacian_matrix) @ np.asarray(states)
    return np.asarray(-gains * consensus_error)
