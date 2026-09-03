"""Figures for one hierarchical-Bayes fit: sampler trustworthiness, then the science.

Split by audience, because the two differ. The diagnostic set decides whether the numbers
may be quoted at all; the reading set is what goes in a talk. Both are logged from inside
the run, so a fit that cost hours never has to be repeated to see it.

What is deliberately NOT here, and why:

``plot_shrinkage``
    Needs an *unpooled* per-subject arm to make its point -- its own docstring says so:
    without it the figure shows where subjects ended up, not that pooling moved them. The
    HB run produces no unpooled estimates, so a faithful version needs a per-subject MLE
    pass (``baseline_rl_hattori``) alongside. Omitted rather than shipped degenerate.

Per-session latent trajectories
    Need the session-level sites, which ``save_fit`` excludes by default, plus a replay of
    the Q recursion. Tracked separately; the likelihood is not reshaped to produce a figure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Order matches HATTORI2019_PARAMS; the last is unbounded and takes no transform.
PARAM_NAMES = (
    "learn_rate_rew",
    "learn_rate_unrew",
    "forget_rate_unchosen",
    "softmax_inverse_temperature",
    "bias_l",
)


def load_fit(netcdf_path, sample_stats_path=None):
    """Read a saved fit back as an InferenceData, whichever layout it was written in.

    Fits written before the round-trip fix put each group at the netCDF root, where
    ``az.from_netcdf`` returns an EMPTY InferenceData without raising. Handle both so
    figures can be regenerated from any archived fit.
    """
    import arviz as az
    import xarray as xr

    idata = az.from_netcdf(str(netcdf_path))
    if list(idata.groups()):
        return idata

    logger.info("Fit at %s is in the pre-grouped layout; reading groups directly.",
                netcdf_path)
    groups = {"posterior": xr.open_dataset(str(netcdf_path))}
    if sample_stats_path and Path(sample_stats_path).exists():
        groups["sample_stats"] = xr.open_dataset(str(sample_stats_path))
    return az.InferenceData(**groups)


def to_bounded(unconstrained, param_index, beta_max=10.0):
    """Map one parameter from the sampler's scale to the one it is named for."""
    from scipy.stats import norm

    values = np.asarray(unconstrained)
    if param_index == 4:            # side bias is unbounded
        return values
    if param_index == 3:            # inverse temperature
        return norm.cdf(values) * beta_max
    return norm.cdf(values)         # the three rates, on [0, 1]


def plot_sampler_diagnostics(idata, path=None):
    """Trace, rank and energy for the population parameters.

    These are the figures a fit is judged on before its numbers are used: a trace that has
    not mixed, ranks that are not uniform across chains, or an energy pathology all mean the
    posterior is not the posterior.
    """
    import matplotlib
    matplotlib.use("Agg")
    import arviz as az
    import matplotlib.pyplot as plt

    n_chains = int(idata.posterior.sizes.get("chain", 1))
    has_energy = (
        "sample_stats" in idata.groups() and "energy" in idata.sample_stats
    )
    # Rank plots compare chains against each other, so they say nothing at one chain.
    panels = ["trace"] + (["rank"] if n_chains > 1 else []) + (["energy"] if has_energy else [])

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(7.2, 2.6 * len(panels)), squeeze=False
    )
    axes = axes[:, 0]
    for ax, kind in zip(axes, panels):
        if kind == "trace":
            draws = np.asarray(idata.posterior["population_mean"])
            for chain in range(draws.shape[0]):
                for p in range(draws.shape[-1]):
                    ax.plot(draws[chain, :, p], lw=0.7, alpha=0.8)
            ax.set_title(
                f"population_mean traces ({n_chains} chain{'s' if n_chains > 1 else ''})",
                loc="left", fontsize=9,
            )
            ax.set_xlabel("draw")
        elif kind == "rank":
            az.plot_rank(idata, var_names=["population_mean"], ax=ax)
        else:
            az.plot_energy(idata, ax=ax)

    if n_chains == 1:
        fig.text(
            0.01, 0.005,
            "One chain: r-hat is undefined and rank plots are uninformative. "
            "Not a converged fit.",
            fontsize=7, style="italic",
        )
    fig.tight_layout()
    if path:
        fig.savefig(str(path), dpi=170, bbox_inches="tight")
    return fig


def plot_population_posteriors(idata, beta_max=10.0, path=None):
    """Population posterior per parameter, on the parameter's own scale.

    Drawn post-transform because plotting unconstrained coordinates under a bounded
    parameter's label misstates every value.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    draws = np.asarray(idata.posterior["population_mean"]).reshape(-1, len(PARAM_NAMES))
    fig, axes = plt.subplots(
        len(PARAM_NAMES), 1, figsize=(4.6, 1.15 * len(PARAM_NAMES)), squeeze=False
    )
    for i, (ax, name) in enumerate(zip(axes[:, 0], PARAM_NAMES)):
        values = to_bounded(draws[:, i], i, beta_max)
        spread = float(values.max() - values.min())
        if spread > 0 and len(np.unique(values)) > 2:
            kde = gaussian_kde(values)
            grid = np.linspace(values.min() - 0.18 * spread, values.max() + 0.18 * spread, 200)
            ax.fill_between(grid, kde(grid), alpha=0.55, lw=0)
            ax.plot(grid, kde(grid), lw=1.0)
        else:
            ax.axvline(float(values.mean()), lw=1.4)
        ax.set_yticks([])
        ax.set_ylabel(name.replace("_", "\n"), rotation=0, ha="right",
                      va="center", fontsize=6, labelpad=4)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
    axes[-1, 0].set_xlabel("parameter value (model's own units)", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(str(path), dpi=170, bbox_inches="tight")
    return fig


def log_hb_figures(
    fit_paths, scores=None, wandb_run=None, output_dir=None,
    beta_max=10.0, references=None,
):
    """Build the figure set for one fit and log it to W&B.

    Parameters
    ----------
    fit_paths : mapping
        ``{"netcdf": ..., "sample_stats": ...}`` as returned by ``save_fit``.
    scores : mapping, optional
        Context-session count to held-out likelihood, plus an optional ``"matched"`` key.
        Drives the conditioning curve, which is the headline comparison figure.
    wandb_run : optional
        Run to log into. Figures are still written to ``output_dir`` without one.
    references : mapping, optional
        ``{name: (value, colour)}`` comparators for the conditioning curve.

    Returns
    -------
    dict
        Figure key to written path, for the keys that were produced.
    """
    written = {}

    try:
        output_dir = Path(output_dir or ".")
        output_dir.mkdir(parents=True, exist_ok=True)
        idata = load_fit(fit_paths.get("netcdf"), fit_paths.get("sample_stats"))

        specs = [
            ("hb/diagnostics", "hb_diagnostics.png",
             lambda p: plot_sampler_diagnostics(idata, path=p)),
            ("hb/population_posterior", "hb_population_posterior.png",
             lambda p: plot_population_posteriors(idata, beta_max=beta_max, path=p)),
        ]
        if scores:
            from aind_dynamic_foraging_models.hierarchical_bayes.plotting import (
                plot_conditioning_curve,
            )
            specs.append((
                "hb/conditioning_curve", "hb_conditioning_curve.png",
                lambda p: plot_conditioning_curve(
                    scores, references=references, path=p,
                    title="Held-out likelihood vs context sessions",
                ),
            ))
    except Exception:
        logger.exception("Could not prepare HB figures; continuing.")
        return written

    for key, filename, build in specs:
        path = output_dir / filename
        try:
            build(str(path))
            written[key] = str(path)
        except Exception:
            # A figure is never worth losing a completed fit over.
            logger.exception("Could not build %s; continuing.", key)

    if wandb_run is not None and written:
        try:
            import wandb

            wandb_run.log({k: wandb.Image(v) for k, v in written.items()})
            logger.info("Logged %d HB figures: %s", len(written), ", ".join(written))
        except Exception:
            logger.exception("Could not log HB figures to W&B; files are in %s.", output_dir)

    return written
