"""Create a W&B sweep and submit SLURM sweep agents.

This launcher does three things:

1. Reads a sweep YAML (e.g. ``sweeps/scaling_disrnn.yaml``).
2. Injects per-run lineage fields into the sweep's ``command`` list so every
   run started by every agent records where it came from. Injected fields:

       +meta.git_commit    full SHA of HEAD at launch time
       +meta.git_branch    current branch
       +meta.git_dirty     "yes" if the working tree had uncommitted changes
       +meta.sweep_yaml    path of the sweep YAML (relative to repo root)
       +meta.owner         Unix user who launched the sweep
       +meta.launcher_cmd  exact argv used to invoke this launcher
       +meta.mode          "cpu" or "gpu"

   These appear in each W&B run's config under ``meta.*`` and are filterable
   in the Runs/Sweeps UI, so any run can be traced back to the exact code
   and command that produced it without relying on a separate registry file.

3. Creates the sweep via ``wandb sweep`` (using a temp YAML patched with the
   lineage fields) and submits a SLURM array job of agents.

Notes on W&B sweep behavior:
    - When running under a sweep, W&B ignores per-run ``wandb.entity`` and
      ``wandb.project`` overrides; sweep routing comes from the top-level
      ``entity``/``project`` fields in the sweep YAML.
    - Hydra ``+meta.*`` overrides ARE honored on each run, which is why this
      launcher uses them to carry lineage.

Example:
    python -m code.launch_wandb_sweep \
        --sweep-yaml sweeps/scaling_disrnn.yaml \
        --mode cpu \
        --sbatch-extra=--array=0-1
"""

from __future__ import annotations

import argparse
import getpass
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def _parse_sweep_id(stdout: str, stderr: str) -> str | None:
    text = "\n".join([stdout, stderr])

    patterns = [
        r"wandb\s+agent\s+([\w.-]+/[\w.-]+/[\w-]+)",
        r"Created sweep with ID:\s*([\w-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            if "/" not in candidate and pattern.endswith("([\\w-]+)"):
                return None
            return candidate

    return None


def _run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )


def _extract_array_spec(sbatch_extra: str) -> str | None:
    tokens = shlex.split(sbatch_extra)
    for idx, token in enumerate(tokens):
        if token.startswith("--array="):
            return token.split("=", 1)[1]
        if token == "--array" and idx + 1 < len(tokens):
            return tokens[idx + 1]
    return None


def _read_array_spec_from_script(slurm_script: Path) -> str | None:
    for line in slurm_script.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#SBATCH") and "--array=" in stripped:
            return stripped.split("--array=", 1)[1].strip()
    return None


def _count_array_tasks(array_spec: str) -> int:
    spec = array_spec.split("%", 1)[0].strip()
    if not spec:
        return 1

    total = 0
    for chunk in spec.split(","):
        piece = chunk.strip()
        if not piece:
            continue

        if "-" not in piece:
            total += 1
            continue

        range_part, *step_part = piece.split(":")
        start_s, end_s = range_part.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        step = int(step_part[0]) if step_part else 1
        if step <= 0:
            raise ValueError(f"Invalid array step in spec: {array_spec}")
        if end < start:
            raise ValueError(f"Invalid array range in spec: {array_spec}")
        total += (end - start) // step + 1

    return max(total, 1)


def _git_info(repo_root: Path) -> dict[str, str]:
    """Capture lineage info from the git repo at repo_root.

    Returns commit sha, branch, and a 'dirty' flag (yes/no) so each run in
    the sweep records the exact code state that launched it.
    """
    def _run(args: list[str]) -> str:
        try:
            out = subprocess.run(
                args,
                cwd=str(repo_root),
                check=True,
                capture_output=True,
                text=True,
            )
            return out.stdout.strip()
        except Exception:
            return "unknown"

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    dirty = "yes" if status and status != "unknown" else "no"
    return {"commit": commit, "branch": branch, "dirty": dirty}


def _inject_lineage_into_command(
    sweep_cfg: dict,
    lineage: dict[str, str],
) -> dict:
    """Append Hydra +meta.* overrides to the sweep 'command' list.

    Sweep mode ignores per-run wandb.entity/project overrides, but Hydra
    overrides in the command list are still applied to each run's config,
    so each W&B run carries this lineage in its config under `meta.*`.
    """
    cmd = list(sweep_cfg.get("command", []))
    for key, value in lineage.items():
        # `+meta.<key>=<value>` adds the field (won't error if missing in schema).
        # Quote values containing chars that Hydra's override parser rejects
        # (spaces, '=', commas, etc.) so they survive as plain strings.
        sval = str(value)
        if any(c in sval for c in " =,'\"[]{}"):
            # Use single quotes; escape any embedded single quotes.
            escaped = sval.replace("'", "\\'")
            sval = f"'{escaped}'"
        cmd.append(f"+meta.{key}={sval}")
    sweep_cfg["command"] = cmd
    return sweep_cfg


def _estimate_total_grid_runs(sweep_yaml: Path) -> int | None:
    sweep_cfg = yaml.safe_load(sweep_yaml.read_text())
    if not isinstance(sweep_cfg, dict):
        return None
    if str(sweep_cfg.get("method", "")).lower() != "grid":
        return None

    parameters = sweep_cfg.get("parameters", {})
    if not isinstance(parameters, dict) or not parameters:
        return None

    counts: list[int] = []
    for _, param_cfg in parameters.items():
        if not isinstance(param_cfg, dict):
            return None
        values = param_cfg.get("values")
        if not isinstance(values, list) or len(values) == 0:
            return None
        counts.append(len(values))

    return math.prod(counts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create W&B sweep and submit SLURM array agents."
    )
    parser.add_argument(
        "--sweep-yaml",
        default="sweeps/scaling_disrnn.yaml",
        help="Path to W&B sweep YAML file.",
    )
    parser.add_argument(
        "--mode",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Choose CPU or GPU sweep agent script.",
    )
    parser.add_argument(
        "--agent-count",
        type=int,
        default=None,
        help="Override AGENT_COUNT passed to sbatch. If omitted, estimate from sweep size and array size.",
    )
    parser.add_argument(
        "--sbatch-extra",
        default="",
        help="Extra sbatch arguments, e.g. '--array=0-9'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sweep_yaml = (repo_root / args.sweep_yaml).resolve()
    if not sweep_yaml.exists():
        raise FileNotFoundError(f"Sweep YAML not found: {sweep_yaml}")

    slurm_script = repo_root / "job" / f"wandb_sweep_{args.mode}.slurm"
    if not slurm_script.exists():
        raise FileNotFoundError(f"SLURM script not found: {slurm_script}")

    array_spec = _extract_array_spec(args.sbatch_extra)
    if not array_spec:
        array_spec = _read_array_spec_from_script(slurm_script)
    num_array_tasks = _count_array_tasks(array_spec) if array_spec else 1

    computed_agent_count = args.agent_count
    total_grid_runs = _estimate_total_grid_runs(sweep_yaml)
    if computed_agent_count is None and total_grid_runs is not None:
        computed_agent_count = max(1, math.ceil(total_grid_runs / num_array_tasks))
        print(
            "Computed AGENT_COUNT="
            f"{computed_agent_count} from total_grid_runs={total_grid_runs} and num_array_tasks={num_array_tasks}"
        )
    elif computed_agent_count is None:
        # fallback to script defaults if sweep size cannot be inferred
        computed_agent_count = 3 if args.mode == "cpu" else 15
        print(
            "Could not infer total sweep runs; using fallback AGENT_COUNT="
            f"{computed_agent_count} for mode={args.mode}"
        )

    # Report planned coverage so the user can spot under-sampling early.
    # Each W&B agent runs `AGENT_COUNT` runs, so planned runs is the
    # product of array tasks and the agent count.
    planned_runs = num_array_tasks * computed_agent_count
    if total_grid_runs is not None:
        print(
            f"Planned coverage: {planned_runs} runs "
            f"({num_array_tasks} array tasks x {computed_agent_count} agents) "
            f"out of {total_grid_runs} grid points."
        )
        if planned_runs < total_grid_runs:
            print(
                f"WARNING: planned_runs ({planned_runs}) < total_grid_runs "
                f"({total_grid_runs}); only a subset of the grid will be sampled."
            )

    sweep_cmd = ["wandb", "sweep", str(sweep_yaml)]
    print("Running:", " ".join(shlex.quote(x) for x in sweep_cmd))

    if args.dry_run:
        print("Dry-run mode enabled; skipping sweep creation and sbatch submit.")
        return

    # --- Lineage injection ---
    # Build a temp sweep YAML that appends Hydra +meta.* overrides to the
    # command list, so every run in this sweep carries lineage in its W&B
    # config (filterable in the Sweeps/Runs view).
    git = _git_info(repo_root)
    launcher_cmd = " ".join(shlex.quote(a) for a in sys.argv)
    lineage = {
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "git_dirty": git["dirty"],
        "sweep_yaml": str(sweep_yaml.relative_to(repo_root)),
        "owner": getpass.getuser(),
        "launcher_cmd": launcher_cmd,
        "mode": args.mode,
    }
    sweep_cfg = yaml.safe_load(sweep_yaml.read_text())
    sweep_cfg = _inject_lineage_into_command(sweep_cfg, lineage)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="sweep_", delete=False
    ) as tmp:
        yaml.safe_dump(sweep_cfg, tmp, sort_keys=False)
        patched_yaml = Path(tmp.name)
    print(f"Patched sweep YAML with lineage at: {patched_yaml}")

    sweep_cmd = ["wandb", "sweep", str(patched_yaml)]
    sweep_result = _run_command(sweep_cmd, cwd=repo_root)
    if sweep_result.stdout:
        print(sweep_result.stdout, end="")
    if sweep_result.stderr:
        print(sweep_result.stderr, end="")
    if sweep_result.returncode != 0:
        raise RuntimeError("wandb sweep command failed")

    sweep_id = _parse_sweep_id(sweep_result.stdout, sweep_result.stderr)
    if not sweep_id:
        raise RuntimeError(
            "Could not parse SWEEP_ID from wandb output. "
            "Please run `wandb sweep` manually and submit sbatch with that ID."
        )

    print(f"Parsed SWEEP_ID: {sweep_id}")

    sbatch_cmd: list[str] = ["sbatch"]
    sbatch_cmd.extend(["--export", f"ALL,AGENT_COUNT={computed_agent_count}"])
    if args.sbatch_extra:
        sbatch_cmd.extend(shlex.split(args.sbatch_extra))
    sbatch_cmd.extend([str(slurm_script), sweep_id])

    print("Running:", " ".join(shlex.quote(x) for x in sbatch_cmd))
    sbatch_result = _run_command(sbatch_cmd, cwd=repo_root)
    if sbatch_result.stdout:
        print(sbatch_result.stdout, end="")
    if sbatch_result.stderr:
        print(sbatch_result.stderr, end="")
    if sbatch_result.returncode != 0:
        raise RuntimeError("sbatch command failed")

    # If the user bounded the sweep below the full grid, the W&B sweep will
    # stay in 'Running' state forever (no agent ever marks it finished). Tell
    # the user how to stop it manually after their bounded runs complete.
    if total_grid_runs is not None and planned_runs < total_grid_runs:
        print(
            "\nNote: this is a bounded sweep. After your jobs finish, the W&B "
            "sweep will still show 'Running'. Stop it explicitly with:\n"
            f"    wandb sweep --stop {sweep_id}"
        )


if __name__ == "__main__":
    main()
