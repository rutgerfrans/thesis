#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Error: conda command not found. Make sure Anaconda/Miniconda is installed and in your PATH." >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH:-}"

INITIAL_COMM=$(python3 - <<EOF
import config; print(config.COMM)
EOF
)
echo "Starting experiments with COMM=\"${INITIAL_COMM}\""

LOG_DIR="${SCRIPT_DIR}/logs_full"
mkdir -p "${LOG_DIR}"

SUMMARY_CSV="${SCRIPT_DIR}/logs_full/summary.csv"
echo "trial,workers,architecture,dataset_size,fault_prob,duration_sec" > "${SUMMARY_CSV}"

# Function: run a single trial, record high-precision timing, log output and summary
run_trial() {
  local W=$1 ARCH=$2 DS=$3 FP=$4 TR=$5

  export N_PARTITIONS="${W}"
  export NETWORK_ARCHITECTURE="${ARCH}"
  export TRAIN_SAMPLE_SIZE="${DS}"
  export FAULT_P="${FP}"

  case "${INITIAL_COMM}" in
    tensorflow)
      echo "--> Activating TensorFlow CPU environment (tf_cpu)"
      conda activate tf_cpu
      ;;
    pytorch|sam)
      echo "--> Activating MPI environment for PyTorch/SAM (mpi_env_test)"
      conda activate mpi_env_test
      ;;
    *)
      echo "Error: Unsupported COMM=\"${INITIAL_COMM}\"" >&2
      exit 1
      ;;
  esac

  local ARCH_TAG=${ARCH//,/-}
  local FP_PCT=$(echo "${FP} * 100" | bc | cut -d'.' -f1)
  local FP_TAG; printf -v FP_TAG "%03d" "${FP_PCT}"
  local TS; TS=$(date +%Y%m%d_%H%M%S)
  local LOGFILE="${LOG_DIR}/w${W}_arch${ARCH_TAG}_ds${DS}_fp${FP_TAG}_trial${TR}_${TS}.log"

  echo
  echo "→ ${CURRENT_EXP} trial ${TR}/${TRIALS_THIS}, W=${W}, ARCH=${ARCH}, DS=${DS}, FP=${FP}  →  ${LOGFILE}"

  local START=$(date +%s.%N)
  bash "${SCRIPT_DIR}/run_mnist.sh" > "${LOGFILE}" 2>&1

  local END=$(date +%s.%N)
  local DURATION=$(awk "BEGIN { printf \"%.6f\", ${END} - ${START} }")
  DURATION=${DURATION//,/.}

  echo "${TR},${W},\"${ARCH}\",${DS},${FP},${DURATION}" >> "${SUMMARY_CSV}"
}

# Define the experiments and their parameter sets
EXP_TYPES=(fault_prob)
declare -A WORKERS ARCHS DATA_SIZES FAULT_PS TRIALS

# Scaling workers
WORKERS[workers]="1 2 4 8 12"
ARCHS[workers]="784,16,16,10"
DATA_SIZES[workers]="60000"
FAULT_PS[workers]="0"
TRIALS[workers]=1

# Scaling dataset size
WORKERS[dataset]="4"
ARCHS[dataset]="784,16,16,10"
DATA_SIZES[dataset]="10000 30000 60000"
FAULT_PS[dataset]="0"
TRIALS[dataset]=1

# Horizontal model scaling
WORKERS[horiz_model]="4"
ARCHS[horiz_model]="784,16,16,10 784,32,32,10 784,64,64,10 784,128,128,10 784,256,256,10"
DATA_SIZES[horiz_model]="60000"
FAULT_PS[horiz_model]="0"
TRIALS[horiz_model]=1

# Vertical model scaling
WORKERS[vert_model]="4"
ARCHS[vert_model]="784,16,16,10 784,16,16,16,10 784,16,16,16,16,10 784,16,16,16,16,16,10"
DATA_SIZES[vert_model]="60000"
FAULT_PS[vert_model]="0"
TRIALS[vert_model]=1

# Fault tolerance probability sweep
WORKERS[fault_prob]="4"
ARCHS[fault_prob]="784,16,16,10"
DATA_SIZES[fault_prob]="60000"
FAULT_PS[fault_prob]="0.15 0.20 0.25 0.50 0.75"
TRIALS[fault_prob]=20

# Sweep over experiments
for EXP in "${EXP_TYPES[@]}"; do
  CURRENT_EXP=$EXP
  echo
  echo "=== Starting experiment: ${CURRENT_EXP} ==="
  for W in ${WORKERS[$EXP]}; do
    for ARCH in ${ARCHS[$EXP]}; do
      for DS in ${DATA_SIZES[$EXP]}; do
        for FP in ${FAULT_PS[$EXP]}; do
          TRIALS_THIS=${TRIALS[$EXP]}
          for TR in $(seq 1 "${TRIALS_THIS}"); do
            run_trial "$W" "$ARCH" "$DS" "$FP" "$TR"
          done
        done
      done
    done
  done
done

echo
 echo "All sweeps complete. Logs in ${LOG_DIR}/ ; summary in ${SUMMARY_CSV}."
