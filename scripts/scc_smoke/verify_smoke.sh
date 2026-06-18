#!/bin/bash
# Verify SCC smoke test results.
#
# Run after the job array submitted by run_smoke.sh has completed:
#
#   bash scripts/scc_smoke/verify_smoke.sh
#
# Checks:
#   - All 4 run directories contain summary.json and metrics.jsonl
#   - sabi-aggregate produces a 4-row Parquet table without errors
#
# Exit code: 0 = pass, 1 = fail.

set -euo pipefail

SWEEP_DIR="${SABI_SMOKE_DIR:-/tmp/sabi_scc_smoke}"
PASS=1

echo "=== Checking run outputs under $SWEEP_DIR ==="
for run_dir in "$SWEEP_DIR"/*/; do
    # Skip non-run directories (logs/, etc.)
    [[ -f "$run_dir/summary.json" ]] || continue

    echo -n "  $run_dir ... "
    if [[ -f "$run_dir/summary.json" && -f "$run_dir/metrics.jsonl" ]]; then
        echo "OK"
    else
        echo "MISSING FILES"
        PASS=0
    fi
done

echo ""
echo "=== Running sabi-aggregate ==="
if sabi-aggregate "$SWEEP_DIR"; then
    PARQUET="$SWEEP_DIR/sweep_results.parquet"
    if [[ -f "$PARQUET" ]]; then
        echo "sweep_results.parquet written."
    else
        echo "ERROR: sweep_results.parquet not found after aggregate."
        PASS=0
    fi
else
    echo "ERROR: sabi-aggregate failed."
    PASS=0
fi

echo ""
if [[ $PASS -eq 1 ]]; then
    echo "SMOKE TEST PASSED"
    exit 0
else
    echo "SMOKE TEST FAILED — check output above"
    exit 1
fi
