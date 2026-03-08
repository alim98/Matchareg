#!/bin/bash
# Run all diagnostic tests (A→E) sequentially, logging output.
# Usage: bash scripts/run_diagnostics.sh [--pair N]
#
# Run from the repo root (parent of pipeline/):
#   cd /u/almik/feb25 && bash pipeline/scripts/run_diagnostics.sh

PAIR="${2:-0}"
if [[ "$1" == "--pair" ]]; then
    PAIR="$2"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="pipeline/output/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/diagnostics_${TIMESTAMP}.log"

echo "=== Diagnostic Tests — $(date) ===" | tee "$LOGFILE"
echo "Pair: $PAIR" | tee -a "$LOGFILE"
echo "Log:  $LOGFILE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

TESTS=(
    "A|test_A_evaluation|python -m pipeline.tests.test_A_evaluation --pair $PAIR"
    "B|test_B_fitter_gt|python -m pipeline.tests.test_B_fitter_gt --pair $PAIR --device cuda"
    "C|test_C_features|python -m pipeline.tests.test_C_features --pair $PAIR"
    "D|test_D_matching|python -m pipeline.tests.test_D_matching --pair $PAIR"
    "E|test_E_synthetic|python -m pipeline.tests.test_E_synthetic --pair $PAIR"
)

PASS=0
FAIL=0
RESULTS=()

for entry in "${TESTS[@]}"; do
    IFS='|' read -r label name cmd <<< "$entry"

    echo "===== TEST $label: $name =====" | tee -a "$LOGFILE"
    echo "Command: $cmd" | tee -a "$LOGFILE"
    echo "Started: $(date)" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"

    eval "$cmd" >> "$LOGFILE" 2>&1
    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        STATUS="PASS"
        ((PASS++))
    else
        STATUS="FAIL (exit $EXIT_CODE)"
        ((FAIL++))
    fi

    echo "" | tee -a "$LOGFILE"
    echo ">> Test $label: $STATUS" | tee -a "$LOGFILE"
    echo "Finished: $(date)" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"

    RESULTS+=("Test $label ($name): $STATUS")
done

echo "========== SUMMARY ==========" | tee -a "$LOGFILE"
for r in "${RESULTS[@]}"; do
    echo "  $r" | tee -a "$LOGFILE"
done
echo "" | tee -a "$LOGFILE"
echo "Passed: $PASS / $((PASS + FAIL))" | tee -a "$LOGFILE"
echo "Full log: $LOGFILE" | tee -a "$LOGFILE"
