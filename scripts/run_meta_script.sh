#!/usr/bin/env bash
set -euo pipefail

# Directory for all logs and summary
LOG_DIR=logs_full
mkdir -p "${LOG_DIR}"

# CSV summary file
SUMMARY_CSV="${LOG_DIR}/summary.csv"
echo "trial,workers,architecture,dataset_size,fault_prob,duration_sec" > "${SUMMARY_CSV}"

#--------------------------------------------------------------------------------
# Function: run a single trial, record high-precision timing, log output and summary
#--------------------------------------------------------------------------------
run_trial() {
  local W=$1 ARCH=$2 DS=$3 FP=$4 TR=$5

  # Export environment variables for your training script
  export N_PARTITIONS="${W}"
  export NETWORK_ARCHITECTURE="${ARCH}"
  export TRAIN_SAMPLE_SIZE="${DS}"
  export FAULT_P="${FP}"

  # Prepare tags and logfile name
  local ARCH_TAG=${ARCH//,/-}
  local FP_PCT
  FP_PCT=$(echo "$FP * 100" | bc -l | cut -d'.' -f1)
  local FP_TAG
  printf -v FP_TAG "%03d" "$FP_PCT"
  local TS
  TS=$(date +%Y%m%d_%H%M%S)
  local LOGFILE="${LOG_DIR}/w${W}_arch${ARCH_TAG}_ds${DS}_fp${FP_TAG}_trial${TR}_${TS}.log"

  echo
  echo "→ ${CURRENT_EXP} trial ${TR}/${TRIALS_THIS}, W=${W}, ARCH=${ARCH}, DS=${DS}, FP=${FP}  →  ${LOGFILE}"

  # High-precision timing
  local START END DURATION
  START=$(date +%s.%N)

  # Run the actual training script
  bash run_mnist.sh > "${LOGFILE}" 2>&1

  END=$(date +%s.%N)
  DURATION=$(awk "BEGIN { printf \"%.6f\", ${END} - ${START} }")

  # Append to summary CSV
  echo "${TR},${W},\"${ARCH}\",${DS},${FP},${DURATION}" >> "${SUMMARY_CSV}"
}

#--------------------------------------------------------------------------------
# Define the five experiments and their parameter sets
#--------------------------------------------------------------------------------
EXP_TYPES=(workers)
#EXP_TYPES=(workers dataset horiz_model vert_model fault_prob)

declare -A WORKERS ARCHS DATA_SIZES FAULT_PS

# 1) Scaling workers
WORKERS[workers]="32"
ARCHS[workers]="784,16,16,10"
DATA_SIZES[workers]="60000"
FAULT_PS[workers]="0"

# 2) Scaling dataset size
WORKERS[dataset]="4"
ARCHS[dataset]="784,16,16,10"
DATA_SIZES[dataset]="10000 30000 60000"
FAULT_PS[dataset]="0"

# 3) Horizontal model scaling
WORKERS[horiz_model]="4"
ARCHS[horiz_model]="784,16,16,10 784,32,32,10 784,64,64,10 784,128,128,10 784,256,256,10"
DATA_SIZES[horiz_model]="60000"
FAULT_PS[horiz_model]="0"

# 4) Vertical model scaling
WORKERS[vert_model]="4"
ARCHS[vert_model]="784,16,16,10 784,16,16,16,10 784,16,16,16,16,10 784,16,16,16,16,16,10"
DATA_SIZES[vert_model]="60000"
FAULT_PS[vert_model]="0"

# 5) Fault tolerance probability sweep
WORKERS[fault_prob]="4"
ARCHS[fault_prob]="784,16,16,10"
DATA_SIZES[fault_prob]="60000"
FAULT_PS[fault_prob]="0 0.01 0.05 0.1"

#--------------------------------------------------------------------------------
# Loop over each experiment, each parameter combination, and run trials
#--------------------------------------------------------------------------------
for CURRENT_EXP in "${EXP_TYPES[@]}"; do
  echo
  echo "=== Starting experiment: ${CURRENT_EXP} ==="

  for W in ${WORKERS[$CURRENT_EXP]}; do
    for ARCH in ${ARCHS[$CURRENT_EXP]}; do
      for DS in ${DATA_SIZES[$CURRENT_EXP]}; do
        for FP in ${FAULT_PS[$CURRENT_EXP]}; do

          # If we're in the fault_prob sweep and FP != "0", do 20 trials; else 1
          if [[ "${CURRENT_EXP}" == "fault_prob" ]] && [[ "${FP}" != "0" ]]; then
            TRIALS_THIS=20
          else
            TRIALS_THIS=1
          fi

          for TR in $(seq 1 "${TRIALS_THIS}"); do
            run_trial "$W" "$ARCH" "$DS" "$FP" "$TR"
          done

        done
      done
    done
  done

done

echo
echo "All sweeps complete. Logs in ./${LOG_DIR}/ ; summary in ${SUMMARY_CSV}."
