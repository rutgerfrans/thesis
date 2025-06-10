#!/usr/bin/env bash
set -euo pipefail

WORKERS=(4)
ARCHS=(
  "784,256,256,10"
)

DATA_SIZES=(60000)

FAULT_PS=(0.0)

TRIALS=1

LOG_DIR=logs_full
mkdir -p "${LOG_DIR}"

SUMMARY_CSV="${LOG_DIR}/summary.csv"
echo "trial,workers,architecture,dataset_size,fault_prob,duration_sec" > "${SUMMARY_CSV}"

for W in "${WORKERS[@]}"; do
  for ARCH in "${ARCHS[@]}"; do
    for DS in "${DATA_SIZES[@]}"; do
      for FP in "${FAULT_PS[@]}"; do
        for TR in $(seq 1 "${TRIALS}"); do

          export N_PARTITIONS="${W}"
          export NETWORK_ARCHITECTURE="${ARCH}"
          export TRAIN_SAMPLE_SIZE="${DS}"
          export FAULT_P="${FP}"

          TS=$(date +%Y%m%d_%H%M%S)
          ARCH_TAG=${ARCH//,/-}
          FP_TAG=$(printf "%03d" "${FP#0.}")
          LOG="${LOG_DIR}/w${W}_arch${ARCH_TAG}_ds${DS}_fp${FP_TAG}_trial${TR}_${TS}.log"

          echo
          echo "→ Trial ${TR}: Workers=${W}, Arch=${ARCH}, Dataset=${DS}, Fault=${FP}"
          echo "  Logging to: ${LOG}"

          START_TIME=$(date +%s.%N)
          bash run_mnist.sh > "${LOG}" 2>&1
          END_TIME=$(date +%s.%N)
          DURATION=$(echo "${END_TIME} - ${START_TIME}" | bc)

          echo "${TR},${W},\"${ARCH}\",${DS},${FP},${DURATION}" >> "${SUMMARY_CSV}"

        done
      done
    done
  done
done

echo "Sweep complete. Logs in ./${LOG_DIR}/; summary in ${SUMMARY_CSV}."
