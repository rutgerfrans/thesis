from syndicate import relay, Record, patterns as P, dataspace, turn, Embedded
from syndicate.actor import Facet
from syndicate.during import During
import config as cfg
from src.data_loader import load_dataset, serialize_data, deserialize_data
import src.mnist as mnist

TrainingJob = Record.makeConstructor('TJob', 'gm pid k')
Job          = Record.makeConstructor('job', 'spec')
ExitServer   = Record.makeConstructor('exit', 'code')
Partitions   = Record.makeConstructor('Partitions', 'data')

NUM_EPOCHS = cfg.N_EPOCHS

@relay.service(name='epoch-manager')
@During().add_handler
def main(data):
    control = data['syndicate-server-control'].embeddedValue
    ds = data['worker-dataspace'].embeddedValue

    train_partitions, test_set = load_dataset()
    turn.log.info("asserting partitions.")
    turn.publish(ds, Partitions([serialize_data(p) for p in train_partitions]))

    if cfg.TRAIN_SAMPLE_SIZE > 0:
        per_worker = cfg.TRAIN_SAMPLE_SIZE // cfg.N_PARTITIONS
        train_partitions = [p[:per_worker] for p in train_partitions]

    def run_epoch(global_model, current_epoch):
        if current_epoch > NUM_EPOCHS:
            turn.log.info("All %d epochs finished!", NUM_EPOCHS)
            turn.send(control, ExitServer(0))
            return

        @turn.facet
        def the_facet_body():
            collected_results = []
            epoch_facet = Facet.active
            @During().add_handler
            def k(local_model):
                local_model = deserialize_data(local_model)
                collected_results.append(local_model)
                if len(collected_results) == len(train_partitions):
                    updated_global_model = mnist.update_model(collected_results,train_partitions)
                    turn.log.info(f"Epoch: {current_epoch} / {cfg.N_EPOCHS} | "f"Final Model Evaluation: {updated_global_model.evaluate(test_set)} / {len(test_set)}")
                    turn.stop(facet = epoch_facet,continuation = lambda: run_epoch(updated_global_model,current_epoch + 1))

            for pid in range(len(train_partitions)):
                turn.publish(ds,Job(TrainingJob(serialize_data(global_model),pid,Embedded(turn.ref(k)))))

    run_epoch(mnist.Network(cfg.NETWORK_ARCHITECTURE), 1)
