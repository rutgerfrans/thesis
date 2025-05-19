from syndicate import relay, Record, turn
from syndicate.during import During
import src.utils as utils
from src.comms.comm import BaseFederatedCommunicator
import src.mnist as mnist

TrainingJob = Record.makeConstructor('TJob', 'gm tp k') # global_model, training_partition

@relay.service(name='worker')
@During().add_handler
def main(data):
    comm = BaseFederatedCommunicator()

    if TrainingJob.isClassOf(data):
        #turn.log.info("received training job, starting training now.")
        gm = data[0]
        pt = data[1]
        local_model = comm.train_model(gm, utils.load_pickle(pt))
        turn.publish(TrainingJob._k(data).embeddedValue, [local_model, pt])