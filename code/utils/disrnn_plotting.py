"""Reusable plotting helpers for disRNN evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def _ensure_1d(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={out.shape}")
    return out


def plot_latents_over_trials(
    choices: np.ndarray,
    rewards: np.ndarray,
    latents: np.ndarray,
    *,
    open_latents: Sequence[int] | None = None,
    action_probabilities: np.ndarray,
    fig_dpi: int = 100,
) -> plt.Figure:
    """Plot choice/reward history and latent trajectories across trials.

    Parameters
    ----------
    choices:
        1D array of previous choices for one session.
    rewards:
        1D array of previous rewards for one session.
    latents:
        2D array ``[n_trials, n_latents]`` for one session.
    open_latents:
        Which latent indices to show in the middle panel.
    action_probabilities:
        2D array ``[n_trials, n_actions]`` containing model-predicted action
        probabilities for one session.
    fig_dpi:
        Figure resolution.
    """
    choices = _ensure_1d(choices)
    rewards = _ensure_1d(rewards)
    latents = np.asarray(latents)
    action_probabilities = np.asarray(action_probabilities)
    if latents.ndim != 2:
        raise ValueError(f"Expected latents to be 2D, got shape={latents.shape}")
    if action_probabilities.ndim != 2:
        raise ValueError(
            "Expected action_probabilities to be 2D, "
            f"got shape={action_probabilities.shape}"
        )

    if (
        choices.shape[0] != rewards.shape[0]
        or choices.shape[0] != latents.shape[0]
        or choices.shape[0] != action_probabilities.shape[0]
    ):
        raise ValueError(
            "choices, rewards, latents, and action_probabilities must have "
            "the same n_trials"
        )

    if open_latents is None:
        open_latents = tuple(range(latents.shape[1]))

    valid_mask = (choices != -1) & (rewards != -1)
    valid_indices = np.where(valid_mask)[0]
    choices_filtered = choices[valid_mask]
    rewards_filtered = rewards[valid_mask]
    latents_filtered = latents[valid_mask]
    action_probabilities_filtered = action_probabilities[valid_mask]

    fig, axs = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 10),
        dpi=fig_dpi,
        sharex=True,
    )

    ax = axs[0]
    choice_mask = (choices_filtered == 0) | (choices_filtered == 1)
    non_choice_mask = ~choice_mask
    rewarded_mask = (rewards_filtered == 1) & choice_mask
    unrewarded_mask = (rewards_filtered == 0) & choice_mask

    ax.scatter(
        valid_indices[non_choice_mask],
        choices_filtered[non_choice_mask],
        marker="|",
        s=800,
        color="gray",
        label="No choice",
        alpha=0.7,
    )
    ax.scatter(
        valid_indices[unrewarded_mask],
        choices_filtered[unrewarded_mask],
        marker="|",
        s=400,
        color="black",
        label="Unrewarded",
        alpha=0.7,
    )
    ax.scatter(
        valid_indices[rewarded_mask],
        choices_filtered[rewarded_mask],
        marker="|",
        s=800,
        color="black",
        label="Rewarded",
        alpha=0.7,
    )
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Left", "Right"])
    ax.set_title("Choices history", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=12)

    ax = axs[1]
    for latent_idx in open_latents:
        if 0 <= latent_idx < latents_filtered.shape[1]:
            ax.plot(valid_indices, latents_filtered[:, latent_idx], label=f"latent {latent_idx}")
    ax.set_title("Open latents", fontsize=18)
    ax.set_xlabel("Trial", fontsize=18)
    ax.set_ylabel("Latent value", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=12)

    ax = axs[2]
    for action_idx in range(action_probabilities_filtered.shape[1]):
        label = f"Action {action_idx}"
        if action_probabilities_filtered.shape[1] == 2:
            label = "P(Left)" if action_idx == 0 else "P(Right)"
        ax.plot(valid_indices, action_probabilities_filtered[:, action_idx], label=label)
    ax.set_title("Predicted action probabilities", fontsize=18)
    ax.set_xlabel("Trial", fontsize=18)
    ax.set_ylabel("Probability", fontsize=18)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=12)

    fig.tight_layout()
    return fig


def plot_latents_in_space(
    latent_states: np.ndarray,
    *,
    color_values: np.ndarray,
    color_label: str,
    selected_latents: Sequence[int] | None = None,
    example_run: np.ndarray | None = None,
    axis_label_prefix: str = "Latent",
    color_minmax: tuple[float, float] = (0.0, 1.0),
    fig_dpi: int = 100,
) -> plt.Figure:
    """Plot pairwise latent-space trajectories for selected latent dimensions.

    Parameters
    ----------
    latent_states:
        Array with shape ``[..., n_latents]`` where the last axis is latent
        dimension and all leading axes are flattened into points.
    color_values:
        Array with shape matching ``latent_states.shape[:-1]`` (or already
        flattened to ``[n_points]``) used for background point coloring.
    color_label:
        Label for the colorbar.
    selected_latents:
        Latent indices to include in the pairwise plot grid.
    example_run:
        Optional array ``[n_trials, n_latents_selected]`` to overlay as a trajectory.
    axis_label_prefix:
        Prefix used for axis labels, e.g. ``Latent`` or ``PC``.
    color_minmax:
        Tuple ``(vmin, vmax)`` for color scaling.
    fig_dpi:
        Figure resolution.
    """
    latent_states = np.asarray(latent_states)
    color_values = np.asarray(color_values)

    if latent_states.ndim < 2:
        raise ValueError(
            "Expected latent_states with latent axis in the last dimension, "
            f"got shape={latent_states.shape}"
        )

    n_latents = latent_states.shape[-1]
    latent_points = latent_states.reshape(-1, n_latents)

    if color_values.shape == latent_states.shape[:-1]:
        color_points = color_values.reshape(-1)
    elif color_values.ndim == 1:
        color_points = color_values
    else:
        raise ValueError(
            "Expected color_values to match latent_states leading dimensions "
            f"{latent_states.shape[:-1]} or be 1D, got shape={color_values.shape}"
        )

    if color_points.shape[0] != latent_points.shape[0]:
        raise ValueError(
            "color_values must match latent_states point count. "
            f"Got colors={color_points.shape[0]} points={latent_points.shape[0]}"
        )

    if selected_latents is None:
        selected_latents = tuple(range(n_latents))
    selected_latents = tuple(int(i) for i in selected_latents if 0 <= int(i) < n_latents)
    if len(selected_latents) < 2:
        raise ValueError("Need at least two latent dimensions to plot latent space")

    num_dims = len(selected_latents)
    n_axes = num_dims * (num_dims - 1) / 2
    n_cols = num_dims
    n_rows = int(n_axes / n_cols) + (n_axes % n_cols > 0)
    if n_rows == 1:
        n_rows = 2

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        dpi=fig_dpi,
    )
    axs = np.atleast_2d(axs)

    ax_id = 0
    for i, latent_x in enumerate(selected_latents):
        for j, latent_y in enumerate(selected_latents):
            if i < j:
                ax = axs[int(ax_id / n_cols), ax_id % n_cols]
                scatter = ax.scatter(
                    latent_points[:, latent_x],
                    latent_points[:, latent_y],
                    s=2.5,
                    c=color_points,
                    cmap=cm.coolwarm,
                    vmin=color_minmax[0],
                    vmax=color_minmax[1],
                    alpha=0.9,
                )
                ax.set_title("Latent space trajectory", fontsize=18)
                ax.set_xlabel(f"{axis_label_prefix} {latent_x}", fontsize=16)
                ax.set_ylabel(f"{axis_label_prefix} {latent_y}", fontsize=16)
                ax.tick_params(axis="both", which="major", labelsize=11)
                fig.colorbar(scatter, ax=ax, label=color_label)

                if example_run is not None:
                    ax.plot(
                        example_run[:, i],
                        example_run[:, j],
                        color="k",
                        lw=1,
                        alpha=0.9,
                    )
                    run_len = example_run.shape[0]
                    run_scatter = ax.scatter(
                        example_run[:, i],
                        example_run[:, j],
                        s=8,
                        c=np.arange(run_len),
                        cmap=cm.copper,
                        vmin=0,
                        vmax=run_len,
                    )
                    fig.colorbar(run_scatter, ax=ax, label="Time step")

                ax_id += 1

    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Save and close a matplotlib figure."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
