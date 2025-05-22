#!/bin/bash
cd ..

COMM=$(python3 -c "import config; print(config.COMM)")
NETWORK_ARCHITECTURE=$(python3 -c "import config; print(config.NETWORK_ARCHITECTURE)")
N_EPOCHS=$(python3 -c "import config; print(config.N_EPOCHS)")
N_PARTITIONS=$(python3 -c "import config; print(config.N_PARTITIONS)")
DEBUG=$(python3 -c "import config; print(config.DEBUG)")

echo "==============================="
echo "Running MNIST"
echo "Comm Prot: $COMM"
echo "NN ARCH: $NETWORK_ARCHITECTURE"
echo "N_EPOCHS: $N_EPOCHS"
echo "N_PARTITIONS: $N_PARTITIONS"
echo "==============================="

if [ "$COMM" == "mpi" ]; then
    mpiexec -n $(( N_PARTITIONS + 1 )) python3 -m src.mpi.driver

elif [ "$COMM" == "sam" ]; then 
    syndicate-server -c config/rpc-syndicate-config.pr

elif [ "$COMM" == "pytorch" ]; then 
    # Automatic restart loop on failure
    RETRIES=0
    while true; do
        echo "[Attempt $((RETRIES+1))] Starting PyTorch federated training..."
        torchrun --nnodes 1 --nproc_per_node "$N_PARTITIONS" --max_restart 0 -m src.pytorch.main
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Training completed successfully."
            break
        fi
        RETRIES=$((RETRIES+1))
        echo "Training failed with exit code $EXIT_CODE."
        echo "Restarting in 10 seconds..."
        sleep 10
    done

elif [ "$COMM" == "tensorflow" ]; then
    HOST=localhost
    BASE_PORT=12345
    WORKERS=$(seq $BASE_PORT $((BASE_PORT+N_PARTITIONS-1)) \
              | sed "s/.*/\"$HOST:&\"/" \
              | paste -sd, -)
    for i in $(seq 0 $((N_PARTITIONS-1))); do
      (
        RETRIES=0
        while true; do
          export TF_CONFIG="{\"cluster\":{\"worker\":[$WORKERS]},\"task\":{\"type\":\"worker\",\"index\":$i}}"
          echo "[TF worker $i | attempt $RETRIES] TF_CONFIG=$TF_CONFIG"
          python3 -m src.tensorflow.main
          EXIT_CODE=$?
          if [ $EXIT_CODE -eq 0 ]; then
            echo ">> Worker $i exited normally."
            break
          fi
          RETRIES=$((RETRIES+1))
          echo ">> Worker $i crashed with exit code $EXIT_CODE; restarting in 5s..."
          sleep 5
        done
      ) &
    done
    wait
fi
