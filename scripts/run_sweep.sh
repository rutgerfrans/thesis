#!/usr/bin/env bash
set -euo pipefail

# 1) worker counts to try
WORKERS=(1 2 4 8)

# 2) model architectures (as CSV strings)
#    [784,input-1,...,output]  e.g. “784,32,16,10”
#    you can add as many as you like
ARCHS=(
  "784,16,16,10"
  "784,32,32,10"
  "784,64,64,10"
  "784,128,128,10"
)

# 3) training-set sizes (use -1 for full 60k)
DATA_SIZES=(10000 30000 60000)

LOG_DIR=logs_full
mkdir -p "${LOG_DIR}"

for W in "${WORKERS[@]}"; do
  for ARCH in "${ARCHS[@]}"; do
    for DS in "${DATA_SIZES[@]}"; do

      export N_PARTITIONS=${W}
      export NETWORK_ARCHITECTURE=${ARCH}
      export TRAIN_SAMPLE_SIZE=${DS}

      TS=$(date +%Y%m%d_%H%M%S)
      # sanitize CSV for filename (commas→dashes)
      ARCH_TAG=${ARCH//,/-}
      DS_TAG=${DS}
      LOG="${LOG_DIR}/w${W}_arch${ARCH_TAG}_ds${DS_TAG}_${TS}.log"

      echo
      echo "→ Workers: $W, Arch: $ARCH, TrainSize: $DS"
      echo "  Logging to: $LOG"
      bash run_mnist.sh > "$LOG" 2>&1

    done
  done
done

echo "Sweep complete. Logs in ./${LOG_DIR}/"
