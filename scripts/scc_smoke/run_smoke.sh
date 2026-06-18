#!/bin/bash
# SCC smoke test — submit a minimal 4-run sweep and verify outputs.
#
# Run this script from the repo root on an SCC login node:
#
#   bash scripts/scc_smoke/run_smoke.sh
#
# It will:
#   1. Generate manifest.tsv + qsub_array.sh in /tmp/sabi_scc_smoke/
#   2. Submit the job array via qsub
#   3. Print the job ID so you can watch it with `qstat`
#   4. After the array finishes, run: bash scripts/scc_smoke/verify_smoke.sh
#
# The sweep: 2 problems × 2 seeds, minimum algorithm config (n_initial=4,
# n_rounds=1, prior-sampling acquisition, no metrics).  Total wall-time
# should be under 5 minutes on any modern CPU node.
#
# Prerequisites: sabi installed (uv sync), qsub on PATH.

set -euo pipefail

SWEEP_DIR="${SABI_SMOKE_DIR:-/tmp/sabi_scc_smoke}"

echo "Generating sweep under $SWEEP_DIR ..."
sabi-submit \
    --backend scc_array \
    --output-dir "$SWEEP_DIR" \
    --submit \
    --scc-walltime 00:15:00 \
    --scc-cores 4 \
    --scc-mem-gb 8 \
    --scc-queue shared \
    problem=gaussian_2d,banana \
    acquisition=prior_sampling \
    algorithm.n_initial=4 \
    algorithm.n_rounds=1 \
    metrics=[] \
    --n-seeds 2

echo ""
echo "Job submitted.  Monitor with: qstat"
echo "Once complete, verify with:   bash scripts/scc_smoke/verify_smoke.sh"
echo "SWEEP_DIR=$SWEEP_DIR"
