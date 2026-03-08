#!/usr/bin/env bash
set +e

ROOT="/u/almik/feb25"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT/pipeline/tests/results/stage8_suite"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/stage8_suite_${TS}.log"

PAIR="${PAIR:-0}"
DEVICE="${DEVICE:-cuda}"
DOWNSAMPLE="${DOWNSAMPLE:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_POINTS="${N_POINTS:-4000}"
TARGET_POOL_FEATURE="${TARGET_POOL_FEATURE:-matcha}"

TOTAL=0
FAILED=0

run_test() {
  local name="$1"
  shift
  TOTAL=$((TOTAL + 1))
  echo "================================================================================" | tee -a "$RUN_LOG"
  echo "RUNNING: $name" | tee -a "$RUN_LOG"
  echo "COMMAND: $*" | tee -a "$RUN_LOG"
  echo "================================================================================" | tee -a "$RUN_LOG"
  "$@" 2>&1 | tee -a "$RUN_LOG"
  local code=${PIPESTATUS[0]}
  if [ "$code" -eq 0 ]; then
    echo "RESULT: PASS - $name" | tee -a "$RUN_LOG"
  else
    echo "RESULT: FAIL($code) - $name" | tee -a "$RUN_LOG"
    FAILED=$((FAILED + 1))
  fi
  echo | tee -a "$RUN_LOG"
}

echo "stage8 suite log: $RUN_LOG" | tee -a "$RUN_LOG"
echo "pair=$PAIR device=$DEVICE downsample=$DOWNSAMPLE batch_size=$BATCH_SIZE n_points=$N_POINTS target_pool_feature=$TARGET_POOL_FEATURE" | tee -a "$RUN_LOG"
echo | tee -a "$RUN_LOG"

if [ "$DEVICE" = "cuda" ]; then
  python3 - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  if [ "$?" -ne 0 ]; then
    echo "WARNING: requested DEVICE=cuda but no CUDA/HIP GPU is available; switching to DEVICE=cpu" | tee -a "$RUN_LOG"
    DEVICE="cpu"
    echo | tee -a "$RUN_LOG"
  fi
fi

if [ "$DEVICE" = "cpu" ]; then
  if [ ! -f "$ROOT/pipeline/output/feature_cache/ThoraxCBCT_0000_0001_${TARGET_POOL_FEATURE}.npz" ]; then
    if [ -f "$ROOT/pipeline/output/feature_cache/ThoraxCBCT_0000_0001_dinov3.npz" ]; then
      echo "WARNING: cpu mode without ${TARGET_POOL_FEATURE} cache; switching target_pool_feature=dinov3" | tee -a "$RUN_LOG"
      TARGET_POOL_FEATURE="dinov3"
      echo | tee -a "$RUN_LOG"
    fi
  fi
fi

run_test "test_8_coordinate_sampling_audit" \
  python3 -m pipeline.tests.test_8_coordinate_sampling_audit \
  --pair "$PAIR" --feature all --device "$DEVICE" --downsample "$DOWNSAMPLE" --batch-size "$BATCH_SIZE"

run_test "test_9_backend_feature_audit" \
  python3 -m pipeline.tests.test_9_backend_feature_audit \
  --pair "$PAIR" --feature all --device "$DEVICE" --downsample "$DOWNSAMPLE" \
  --batch-size "$BATCH_SIZE" --n-points "$N_POINTS"

run_test "test_10_backend_matching_audit" \
  python3 -m pipeline.tests.test_10_backend_matching_audit \
  --pair "$PAIR" --feature all --device "$DEVICE" --downsample "$DOWNSAMPLE" --batch-size "$BATCH_SIZE"

run_test "test_11_mind_baseline_audit" \
  python3 -m pipeline.tests.test_11_mind_baseline_audit \
  --pair "$PAIR" --device "$DEVICE"

run_test "test_12_fitter_stress_audit" \
  python3 -m pipeline.tests.test_12_fitter_stress_audit \
  --pair "$PAIR" --device "$DEVICE"

run_test "test_13_sampling_recall_audit" \
  python3 -m pipeline.tests.test_13_sampling_recall_audit \
  --pair "$PAIR" --device "$DEVICE" --n-points 8000 --eval-n 4000

run_test "test_14_target_pool_matchability_audit" \
  python3 -m pipeline.tests.test_14_target_pool_matchability_audit \
  --pair "$PAIR" --feature "$TARGET_POOL_FEATURE" --device "$DEVICE" \
  --downsample "$DOWNSAMPLE" --batch-size "$BATCH_SIZE" --n-points "$N_POINTS"

echo "================================================================================" | tee -a "$RUN_LOG"
echo "STAGE8 SUITE SUMMARY" | tee -a "$RUN_LOG"
echo "TOTAL=$TOTAL FAILED=$FAILED PASSED=$((TOTAL - FAILED))" | tee -a "$RUN_LOG"
echo "LOG=$RUN_LOG" | tee -a "$RUN_LOG"
echo "================================================================================" | tee -a "$RUN_LOG"

exit 0
