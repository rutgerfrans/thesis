#!/usr/bin/env python3

import sys
from syndicate import relay, turn, Symbol, Record, dataspace
from syndicate import patterns as P
from syndicate.during import During
from src.data_loader import load_dataset
from src.comms.ipc_comm import IPCFederatedCommunicator
import src.utils as utils
import config as cfg

Shutdown = Symbol('shutdown')

@relay.service(name='coordinator')
@During().add_handler
def main(config):
    global_model_file = "tmpsam/global_model.pkl"
    comm = IPCFederatedCommunicator()
    train_partitions, test_set = load_dataset()
    
    global_model = comm.init_global_model()
    utils.save_pickle(global_model, global_model_file)

    worker_ds = config['worker-dataspace'].embeddedValue
    #turn.log.info('Got worker-dataspace %s', worker_ds)
    local_models = [] 
    ready_workers = []
    epoch = 0

    @dataspace.during(worker_ds, P.rec('worker-ready', P.CAPTURE, P.CAPTURE))
    def worker_available(w, n):
        nonlocal epoch
        nonlocal ready_workers    
        ready_workers.append(n)
        #turn.log.info('Worker %s is ready', n)
        if epoch < cfg.N_EPOCHS and len(ready_workers) == cfg.N_PARTITIONS:
            # create datastack
            data_stack = comm.create_data_stack(global_model_file, train_partitions)
            # turn.log.info('Distributing training partitions: %s', data_stack)
            # distribute datastack
            handle = turn.publish(worker_ds, Record(Symbol('distribute-datastack'), [data_stack]))
            turn.retract(handle)
            ready_workers = []
            epoch = epoch + 1
        elif epoch >= cfg.N_EPOCHS and len(ready_workers) == cfg.N_PARTITIONS:
            turn.log.info("All %d epochs complete, shutting everything down", cfg.N_EPOCHS)
            turn.publish(worker_ds, Record(Shutdown, []))
            turn.log.info("Sent Shutdown record")
            turn.stop()

    @dataspace.during(worker_ds, P.rec('update-globalmodel', P.CAPTURE))
    def on_localmodel(data):
        nonlocal local_models
        local_models.append(data[0])
        if len(local_models) == len(train_partitions):
            unpickled_models = []
            for model in local_models:
                unpickled_model = utils.load_pickle(model)
                unpickled_models.append(unpickled_model)
            updated_global_model = comm.update_model(unpickled_models, train_partitions, test_set, 0)
            utils.save_pickle(updated_global_model, global_model_file)
            turn.log.info(f"Epoch: {epoch} / {cfg.N_EPOCHS} | Final Model Evaluation: {updated_global_model.evaluate(test_set)} / {len(test_set)}")
            local_models = []
        handle = turn.publish(worker_ds, Record(Symbol('worker-ready'), [data[1], data[2]]))
        turn.retract(handle)