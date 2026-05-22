"""Create a W&B sweep and submit SLURM agents.

Example:
    python -m code.launch_wandb_sweep \
        --sweep-yaml sweeps/scaling_disrnn.yaml \
        --mode cpu
"""

from __future__ import annotations

import argparse
import math
import re
import shlex
import subprocess
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
        help="Choose CPU or GPU agent script.",
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

    slurm_script = repo_root / "job" / f"wandb_agent_{args.mode}.slurm"
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

    sweep_cmd = ["wandb", "sweep", str(sweep_yaml)]
    print("Running:", " ".join(shlex.quote(x) for x in sweep_cmd))

    if args.dry_run:
        print("Dry-run mode enabled; skipping sweep creation and sbatch submit.")
        return

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


if __name__ == "__main__":
    main()
