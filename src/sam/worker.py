from syndicate import relay, Record, turn
from syndicate.during import During
import config as cfg
from src.data_loader import serialize_data, deserialize_data

TrainingJob = Record.makeConstructor('TJob', 'gm tp k')
@relay.service(name='worker')
@During().add_handler
def main(data):
    if TrainingJob.isClassOf(data):
        gm = deserialize_data(TrainingJob._gm(data))
        pt = TrainingJob._tp(data)
        gm.SGD(deserialize_data(pt), cfg.SGD_EPOCHS, cfg.MINI_BATCH_SIZE, cfg.ETA) 
        turn.publish(TrainingJob._k(data).embeddedValue, [serialize_data(gm), pt])