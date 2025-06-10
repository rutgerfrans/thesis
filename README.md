# Thesis 2025 Rutger de Groen
This repo contains an implementation of a data distributed training pipeline with the Syndicated Actor Model, PyTorch, and TensorFlow. Running below commands will reproduce the results mentioned in the thesis report. 

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
To run the SAM setup, you will need to install the latest syndicate-server build:
https://git.syndicate-lang.org/syndicate-lang/syndicate-rs/actions/runs/82 

## Run a Setup
```
cd scripts
COMM="sam" ./run_mnist.sh
```

Alter any environment variable to your likings in the config/config.py file or within the command line.

## Run a Sweep
First predifine a set of environment variables in run_sweep.sh to run as a whole sweep. An example:
```
WORKERS=(1 2 4)

ARCHS=(
  "784,16,16,10"
  "784,32,32,10"
  "784,64,64,10"
)

DATA_SIZES=(30000 60000)

FAULT_PS=(0.0 0.01)

TRIALS=5
```

```
cd scripts
./run_sweep.sh
```

## Generate Plots
Note that these values are hardcoded, data can be foun under logs_full/expresults.csv
```
cd scripts
./run_plots.sh
```

## Raw Experiment Data
the raw and process experiment data can be found under /scripts/logs_full/experiments_results.ods