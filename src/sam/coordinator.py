#!/usr/bin/env python3

from syndicate import relay, turn, Symbol, Record, dataspace
from syndicate import patterns as P
from syndicate.during import During
from src.data_loader import load_dataset
from src.comms.ipc_comm import IPCFederatedCommunicator
import src.utils as utils
import config as cfg

@relay.service(name='coordinator')
@During().add_handler
def main(config):
    global_model_file = "tmpsam/global_model.pkl"
    comm = IPCFederatedCommunicator()
    train_partitions, test_set = load_dataset()
    
    global_model = comm.init_global_model()
    utils.save_pickle(global_model, global_model_file)

    worker_ds = config['worker-dataspace'].embeddedValue
    turn.log.info('Got worker_ds %s', worker_ds)

    local_models = [] # not really a fan of this approach

    @dataspace.during(worker_ds, P.rec('worker-ready', P.CAPTURE))
    def worker_available(w):
        # create datastack
        data_stack = comm.create_data_stack(global_model_file, train_partitions)
        turn.log.info('Length of the training partition dict: %s', data_stack)
        
        # distribute datastack
        turn.log.info('Oh a worker is ready %s', w)
        handle = turn.publish(worker_ds, Record(Symbol('Array'), [data_stack]))
        turn.log.info('Sent Helloworld variable to workers')
        #turn.retract(handle)

    @dataspace.during(worker_ds, P.rec('globalmodel', P.CAPTURE))
    def on_globalmodel(gm):
        turn.log.info('Global model has been updated %s', gm)
        local_models.append(gm)
        if len(local_models) == len(train_partitions):
            unpickled_models = []
            for model in local_models:
                unpickled_model = utils.load_pickle(model)
                unpickled_models.append(unpickled_model)
            updated_global_model = comm.update_model(unpickled_models, train_partitions, test_set, 0)
            utils.save_pickle(updated_global_model, global_model_file)
            turn.log.info(f"Final Model Evaluation: {updated_global_model.evaluate(test_set)} / {len(test_set)}")
