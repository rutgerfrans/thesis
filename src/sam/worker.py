from syndicate import relay, Record, turn, patterns as P, dataspace, Embedded
from syndicate.during import During
import config as cfg
from src.data_loader import serialize_data, deserialize_data
import random, os

TrainingJob = Record.makeConstructor('TJob', 'gm pid k')
Partitions  = Record.makeConstructor('Partitions', 'data')
Worker = Record.makeConstructor('worker', 'entity')

@relay.service(name='worker')
@During().add_handler
def main(data):
    ds = data['worker-dataspace'].embeddedValue

    @dataspace.during(ds, P.rec('Partitions', P.CAPTURE))
    def load_part(data):
        turn.log.info('Got partition data')
        local_partitions = [deserialize_data(p) for p in data]

        @During().add_handler
        def process_job(record):
            if(cfg.FAULT_P > 0.0 and random.uniform(0,1) < cfg.FAULT_P):
                turn.log.info(f"Injecting fault")
                os._exit(1)
            pid = TrainingJob._pid(record)
            turn.log.info('Got job, partition %s', pid)
            gm = deserialize_data(TrainingJob._gm(record))
            pt = local_partitions[pid]
            gm.SGD(pt, cfg.SGD_EPOCHS, cfg.MINI_BATCH_SIZE, cfg.ETA)
            handle = TrainingJob._k(record).embeddedValue
            turn.publish(handle, [serialize_data(gm)])

        turn.publish(ds, Worker(Embedded(turn.ref(process_job))))
