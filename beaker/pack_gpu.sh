#!/usr/bin/env bash
# Pack M wandb-agent processes onto ONE GPU (time-slicing).
#
# Our disRNN runs under-fill a big GPU (on an L40s: ~100% util but only ~30%
# power -> host/eval-bound, GPU mostly idle). Running M agents on one GPU overlaps
# one run's host/eval stalls with another's compute, soaking up that headroom.
#
# Invoked (through entrypoint.sh) by the Beaker spec, e.g.:
#   command: [bash, .../entrypoint.sh, bash, .../pack_gpu.sh, "<SWEEP_ID>", "4"]
#
# Usage: pack_gpu.sh <SWEEP_ID> <M> [RUNS_PER_PROC]
#   M               number of agents to pack on this GPU
#   RUNS_PER_PROC   runs each agent does before exiting (default 1)
set -euo pipefail

SWEEP_ID="${1:?usage: pack_gpu.sh <SWEEP_ID> <M> [RUNS_PER_PROC]}"
M="${2:-4}"
RUNS_PER_PROC="${3:-1}"

# JAX preallocates ~75% of the GPU by default, so a 2nd process OOMs. Give each
# process ~1/M of memory (0.9 leaves CUDA-context + framework headroom).
export XLA_PYTHON_CLIENT_MEM_FRACTION="$(python -c "print(round(0.9/$M, 4))")"
echo "[pack_gpu] packing M=$M agents on one GPU | MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION | sweep=$SWEEP_ID"

pids=()
for i in $(seq 1 "$M"); do
    wandb agent --count "$RUNS_PER_PROC" "$SWEEP_ID" &
    pids+=("$!")
    sleep 3   # stagger so the M processes don't all JAX-compile at the same instant
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
echo "[pack_gpu] all $M agents finished (fail=$fail)"
exit "$fail"
