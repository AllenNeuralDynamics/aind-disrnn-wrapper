"""Hardware x batch-size benchmark figure for the synthetic-RL -> disRNN job.

Pulls the benchmark runs from W&B and renders a 2-panel figure:
  left  = time per training step (s, post-warmup) vs batch size
  right = time to reach valid_loss < THRESHOLD (min, post-warmup) vs batch size

All runs share the config (num_sessions=100, n_steps=5000, n_warmup_steps=500,
lr=0.005, beta=0.001, seed=0, eval_every_n=2); only batch size / hardware vary.
CPU and T4 ran on Code Ocean (Feb 2026); L40s and H200 on Beaker (Jun 2026).

NB the warm-up: a penalty term is inactive during the first n_warmup_steps, so
validation loss there is not meaningful (it can dip below threshold spuriously).
We therefore ignore eval points with step < n_warmup_steps for the crossing time.

Caveat: the CPU/T4 (Feb) and L40s/H200 (Jun) runs used different code revisions
(active development); CPU/T4 floor at min val ~0.210 and never reach 0.21, so only
the 0.22 threshold is a fair four-way comparison. Single seed -> the time-to-
threshold curve is somewhat noisy.

Run:  python plot_hw_batch.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENTITY = "AIND-disRNN"
NSTEP = 5000
THRESHOLD = 0.22
BS = [128, 256, 512, 1024, 2048]

# hardware -> {batch_size: W&B run id}
RUNS = {
    "CPU":  {128: "jlx2egp9", 256: "xpnfv6xq", 512: "7n3r4x62",
             1024: "unitjg8a", 2048: "2bvl403t"},
    "T4":   {128: "l5z3dbac", 256: "23e7ndiv", 512: "g9jibgqd",
             1024: "v0d2vmg8", 2048: "biflbgn5"},
    "L40s": {512: "ybzvw1l9", 2048: "1wbc4tko"},
    "H200": {512: "vhm9w9hz", 2048: "w5ix4w78"},
}
PROJ = {"CPU": "han_cpu_gpu_test", "T4": "han_cpu_gpu_test",
        "L40s": "ai_hub_test", "H200": "ai_hub_test"}
LABEL = {"CPU": "CPU (Code Ocean)", "T4": "T4 (Code Ocean)",
         "L40s": "L40s (Beaker)", "H200": "H200 (Beaker)"}
COL = {"CPU": "#777777", "T4": "tab:blue", "L40s": "tab:green", "H200": "tab:red"}
ORDER = ["CPU", "T4", "L40s", "H200"]


def _find(d, target):
    """Recursive lookup of a key anywhere in a nested config dict."""
    if isinstance(d, dict):
        for k, v in d.items():
            if k == target:
                return v
            r = _find(v, target)
            if r is not None:
                return r
    return None


def collect():
    api = wandb.Api()
    sps, ttt = {}, {}   # hw -> bs -> value
    for hw, m in RUNS.items():
        sps[hw], ttt[hw] = {}, {}
        for bs, rid in m.items():
            run = api.run(f"{ENTITY}/{PROJ[hw]}/{rid}")
            s = run.summary
            sps[hw][bs] = (s["elapsed_seconds"] - s["warmup_seconds"]) / NSTEP
            wu = _find(run.config, "n_warmup_steps") or 500
            h = run.history(keys=["valid/loss", "_runtime", "_step"], pandas=True)
            h = h.dropna(subset=["valid/loss"]).sort_values("_step")
            post = h[h["_step"] >= wu]            # ignore warm-up-phase eval
            below = post[post["valid/loss"] < THRESHOLD]
            ttt[hw][bs] = float(below.iloc[0]["_runtime"]) / 60 if len(below) else None
    return sps, ttt


def plot(sps, ttt, out_png):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # left: time per training step
    for hw in ORDER:
        xs = [bs for bs in BS if bs in sps[hw]]
        axL.plot(xs, [sps[hw][bs] for bs in xs], "o-",
                 color=COL[hw], label=LABEL[hw], lw=2, ms=7)
    axL.set_yscale("log")
    axL.set_ylabel("time per training step (s, post-warmup)")
    axL.set_title("Time per training step vs batch size\nnum_sessions=100, seed 0")

    # right: time to reach valid_loss < THRESHOLD
    for hw in ORDER:
        xs = [bs for bs in BS if ttt[hw].get(bs) is not None]
        axR.plot(xs, [ttt[hw][bs] for bs in xs], "o-",
                 color=COL[hw], label=LABEL[hw], lw=2, ms=7)
    axR.set_yscale("log")
    axR.set_ylabel(f"time to valid_loss < {THRESHOLD} (min)")
    axR.set_title(f"Time to reach valid_loss < {THRESHOLD}  (post-warmup)\n"
                  "num_sessions=100, seed 0")

    for ax in (axL, axR):
        ax.set_xscale("log", base=2)
        ax.set_xticks(BS)
        ax.set_xticklabels(BS)
        ax.set_xlabel("batch_size")
        ax.legend(title="hardware")
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    import os
    sps, ttt = collect()
    plot(sps, ttt, os.path.join(os.path.dirname(__file__),
                                "hw_batch_benchmark.png"))
