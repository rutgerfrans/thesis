import subprocess, pickle, os
import config
from comm import BaseFederatedCommunicator

class IPCFederatedCommunicator(BaseFederatedCommunicator):
    def __init__(self, parition_files, test_file):
        self.partition_files = parition_files
        self.test_file = test_file

    def distribute_data(self, init_file=None):
        processes = []
        out_files = []
        for i, pfile in enumerate(self.partition_files):
            out_fname = f"temp/trained_model_{i}.pkl"
            out_files.append(out_fname)
            cmd = ["python3", "-u", "worker.py", "--partition", pfile, "--test", self.test_file, "--output", out_fname]
            if init_file:
                cmd += ["--initial", init_file]
            proc = subprocess.Popen(cmd, stdout=None if config.DEBUG else subprocess.PIPE,
                                         stderr=None if config.DEBUG else subprocess.PIPE,
                                         text=True)
            processes.append((proc, i, out_fname))
        return processes, out_files

    def wait_for_completion(self, processes):
        for proc, i, _ in processes:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0 and config.DEBUG:
                print(f"Error in worker {i}:")
                print("STDOUT:", stdout.decode("utf-8"))
                print("STDERR:", stderr.decode("utf-8"))

    def collect_models(self, out_files):
        models = []
        for fname in out_files:
            if os.path.exists(fname):
                with open(fname, "rb") as f:
                    models.append(pickle.load(f))
        return models

    def cleanup(self, files):
        for f in files:
            if os.path.exists(f):
                os.remove(f)
