"""Hardware x batch-size benchmark figure for the synthetic-RL -> disRNN job.

Pulls the benchmark runs from W&B and renders a 2-panel figure:
  left  = time per training step (s, post-warmup) vs batch size
  right = time to reach valid_loss < THRESHOLD (min, post-warmup) vs batch size

Hardware spans three platforms:
  - Code Ocean   : CPU (4-core), T4
  - AI1 on-premise HPC : CPU (44-core), V100   (s/step only -- see below)
  - Beaker (AI Hub)    : L40s, H100, H200

The Beaker/Code-Ocean runs share the config (num_sessions=100, n_steps=5000,
n_warmup_steps=500, lr=0.005, beta=0.001, seed=0, eval_every_n=2); only batch
size / hardware vary. The HPC runs use a shorter schedule (n_steps=200) on an
older code revision, so their *throughput* (s/step) is comparable but their
*time-to-target* is not -- HPC hardware therefore appears only on the left panel.

NB the warm-up: a penalty term is inactive during the first n_warmup_steps, so
validation loss there is not meaningful (it can dip below threshold spuriously).
We ignore eval points with step < n_warmup_steps for the crossing time.

s/step is (elapsed_seconds - warmup_seconds) / n_steps from the run summary -- the
direct measurement, logged when a run finishes. All runs must be complete.

Run:  python plot_hw_batch.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENTITY = "AIND-disRNN"
THRESHOLD = 0.22
BS = [128, 256, 512, 1024, 2048]

# label -> spec. panels: which panel(s) the series appears on.
HW = [
    dict(label="CPU (Code Ocean)", project="han_cpu_gpu_test",
         color="#999999", panels=("sps", "ttt"),
         runs={128: "jlx2egp9", 256: "xpnfv6xq", 512: "7n3r4x62",
               1024: "unitjg8a", 2048: "2bvl403t"}),
    dict(label="CPU (AI1 on-premise HPC)", project="hpc_test",
         color="#000000", panels=("sps",), runs={512: "ya2n3sx5"}),
    dict(label="T4 (Code Ocean)", project="han_cpu_gpu_test",
         color="tab:blue", panels=("sps", "ttt"),
         runs={128: "l5z3dbac", 256: "23e7ndiv", 512: "g9jibgqd",
               1024: "v0d2vmg8", 2048: "biflbgn5"}),
    dict(label="V100 (AI1 on-premise HPC)", project="hpc_test",
         color="tab:cyan", panels=("sps",), runs={512: "q9acoys6"}),
    dict(label="L40s (Beaker)", project="ai_hub_test",
         color="tab:green", panels=("sps", "ttt"),
         runs={512: "ybzvw1l9", 2048: "1wbc4tko"}),
    dict(label="H100 (Beaker)", project="ai_hub_test",
         color="tab:orange", panels=("sps", "ttt"),
         runs={512: "l9cb2l2c", 2048: "ne6hxgvz"}),
    dict(label="H200 (Beaker)", project="ai_hub_test",
         color="tab:red", panels=("sps", "ttt"),
         runs={512: "vhm9w9hz", 2048: "w5ix4w78"}),
]


def _find(d, target):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == target:
                return v
            r = _find(v, target)
            if r is not None:
                return r
    return None


def _metrics(run):
    """Return (s_per_step, time_to_threshold_min). Requires a finished run."""
    s, c = run.summary, run.config
    nstep = _find(c, "n_steps")
    wu_steps = _find(c, "n_warmup_steps") or 500
    el, ws = s.get("elapsed_seconds"), s.get("warmup_seconds")
    if el is None or ws is None:
        raise RuntimeError(f"run {run.id} not finished (no elapsed/warmup logged)")
    sps = (el - ws) / nstep

    h = run.history(keys=["valid/loss", "_runtime", "_step"], pandas=True)
    h = h.dropna(subset=["valid/loss"]).sort_values("_step")
    post = h[h["_step"] >= wu_steps]              # ignore warm-up-phase eval
    below = post[post["valid/loss"] < THRESHOLD]
    ttt = float(below.iloc[0]["_runtime"]) / 60 if len(below) else None
    return sps, ttt


def collect():
    api = wandb.Api()
    data = {}
    for hw in HW:
        d = {"sps": {}, "ttt": {}}
        for bs, rid in hw["runs"].items():
            sps, ttt = _metrics(api.run(f"{ENTITY}/{hw['project']}/{rid}"))
            d["sps"][bs] = sps
            if "ttt" in hw["panels"] and ttt is not None:
                d["ttt"][bs] = ttt
        data[hw["label"]] = d
    return data


def plot(data, out_png):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    for hw in HW:
        d = data[hw["label"]]
        if "sps" in hw["panels"]:
            xs = [bs for bs in BS if bs in d["sps"]]
            axL.plot(xs, [d["sps"][bs] for bs in xs], "o-",
                     color=hw["color"], label=hw["label"], lw=2, ms=7)
        if "ttt" in hw["panels"]:
            xs = [bs for bs in BS if bs in d["ttt"]]
            axR.plot(xs, [d["ttt"][bs] for bs in xs], "o-",
                     color=hw["color"], label=hw["label"], lw=2, ms=7)

    axL.set_ylabel("time per training step (s, post-warmup)")
    axL.set_title("Time per training step vs batch size\nnum_sessions=100, seed 0")
    axR.set_ylabel(f"time to valid_loss < {THRESHOLD} (min)")
    axR.set_title(f"Time to reach valid_loss < {THRESHOLD}  (post-warmup)\n"
                  "num_sessions=100, seed 0  (Beaker/Code Ocean only)")

    for ax in (axL, axR):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(BS)
        ax.set_xticklabels(BS)
        ax.set_xlabel("batch_size")
        ax.legend(title="hardware", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    import os
    data = collect()
    for label in (h["label"] for h in HW):
        d = data[label]
        print(label, "s/step:", {b: round(v, 4) for b, v in d["sps"].items()},
              "| time-to-0.22:", {b: round(v, 2) for b, v in d["ttt"].items()})
    plot(data, os.path.join(os.path.dirname(__file__),
                            "hw_batch_benchmark.png"))
