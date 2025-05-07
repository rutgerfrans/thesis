#!/usr/bin/env python3

from syndicate import relay, turn, Symbol, Record, dataspace
from syndicate import patterns as P
from syndicate.during import During
import src.utils as utils
from src.comms.comm import BaseFederatedCommunicator


@relay.service(name='worker')
@During().add_handler
def main(config):
    comm = BaseFederatedCommunicator()
    worker_ds = config['worker-dataspace'].embeddedValue
    worker_id = config['worker']
    turn.log.info('Got config %s', config)

    @dataspace.during(worker_ds, P.rec('Array' ,P.CAPTURE), inert_ok=True)
    def on_hello(msg):
        # obtain partition and globalmodel and start training
        files = msg[worker_id]
        partition_file, model_file = files
        turn.log.info('Worker %r extracts data: %s', worker_id ,files)
        global_model = comm.train_model(utils.load_pickle(model_file), utils.load_pickle(partition_file))

        # save local model and assert to worker-dataspace
        local_model_file = "tmpsam/local_model_file_"+str(worker_id)+".pkl"
        utils.save_pickle(global_model, local_model_file)
        handle = turn.publish(worker_ds, Record(Symbol('globalmodel'), [str(local_model_file)]))
        turn.log.info('Worker trained local model')
        #turn.retract(handle)




