import src.utils as utils
from src.comms.comm import BaseFederatedCommunicator
import subprocess

class SAMFederatedCommunicator(BaseFederatedCommunicator):
    def __init__(self):
        self.config_path = "config/syndicate-server.config.pr"
        self.partition_file = "tmp/partition_"
    
    def run(self):
        proc = subprocess.Popen(["syndicate-server","-c", self.config_path])
        proc.wait()

