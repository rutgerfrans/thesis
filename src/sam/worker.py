from syndicate import relay, Record, turn
from syndicate.during import During
import config as cfg
from src.data_loader import serialize_data, deserialize_data
import time, random, os

TrainingJob = Record.makeConstructor('TJob', 'gm tp k')

@relay.service(name='worker')
@During().add_handler
def main(data):
    global GLOBAL_EPOCH

    if TrainingJob.isClassOf(data):
        if(cfg.FAULT_P > 0.0 and random.uniform(0,1) < cfg.FAULT_P):
            turn.log.info(f"Injecting fault")
            os._exit(1)

        sr_time = time.perf_counter()
        gm = deserialize_data(TrainingJob._gm(data))
        er_time = time.perf_counter()
        read_time = er_time - sr_time

        pt = TrainingJob._tp(data)

        scc_time = time.perf_counter()

        sr_time = time.perf_counter()
        pt = deserialize_data(pt)
        er_time = time.perf_counter()
        read_time += (er_time - sr_time)

        gm.SGD(pt,cfg.SGD_EPOCHS,cfg.MINI_BATCH_SIZE,cfg.ETA)

        ecc_time = time.perf_counter()
        comp_time = ecc_time - scc_time

        turn.publish(TrainingJob._k(data).embeddedValue,[serialize_data(gm), [read_time,comp_time]])