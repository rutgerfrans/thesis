import os
_raw = os.getenv("NETWORK_ARCHITECTURE", "784,16,16,10").strip()
if _raw.startswith('[') and _raw.endswith(']'):
    _raw = _raw[1:-1]
NETWORK_ARCHITECTURE = [int(s.strip()) for s in _raw.split(",") if s.strip()]

SGD_EPOCHS       = int(os.getenv("SGD_EPOCHS", 10))
MINI_BATCH_SIZE  = int(os.getenv("MINI_BATCH_SIZE", 32))
ETA              = float(os.getenv("ETA", 3.0))
N_EPOCHS         = int(os.getenv("N_EPOCHS", 4))
N_PARTITIONS     = int(os.getenv("N_PARTITIONS", 2))
TEST_SAMPLE_SIZE = int(os.getenv("TEST_SAMPLE_SIZE", 1000))
FAULT_P          = float(os.getenv("FAULT_P", 0.0))
DEBUG            = os.getenv("DEBUG", "True") == "True"
COMM             = os.getenv("COMM", "pytorch") # [pytorch, sam, tensorflow]
TRAIN_SAMPLE_SIZE = int(os.getenv("TRAIN_SAMPLE_SIZE", -1))