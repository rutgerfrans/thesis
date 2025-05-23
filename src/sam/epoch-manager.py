from syndicate import relay, Record, patterns as P, dataspace, turn, Embedded
from syndicate.actor import Facet
from syndicate.during import During
import config as cfg
from src.data_loader import load_dataset, serialize_data, deserialize_data
import src.mnist as mnist
import os, csv, time

TrainingJob = Record.makeConstructor('TJob', 'gm tp k')
Job = Record.makeConstructor('job', 'spec')

NUM_EPOCHS = cfg.N_EPOCHS

TIMING_CSV = os.path.join(os.getcwd(), "src/sam/timings/epoch_timings.csv")
if not os.path.exists(TIMING_CSV):
    with open(TIMING_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["system_overhead","total_avg_computation_time_per_worker"])

@relay.service(name='epoch-manager')
@During().add_handler
def main(data):
    total_computation_time = 0
    total_systemoverhead_time = 0

    ds = data['worker-dataspace'].embeddedValue
    train_partitions, test_set = load_dataset()

    def run_epoch(global_model, current_epoch):
        if current_epoch > NUM_EPOCHS:
            turn.log.info("All %d epochs finished!", NUM_EPOCHS)
            ttc_end = time.perf_counter()
            time_to_completion = ttc_end - ttc_start
            with open(TIMING_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"{time_to_completion-total_computation_time:.6f}", f"{total_computation_time:.6f}"])
            return

        @turn.facet
        def the_facet_body():
            collected_results = []
            epoch_facet = Facet.active

            @During().add_handler
            def k(local_model, time_to_completion_worker):
                nonlocal total_computation_time
                nonlocal total_systemoverhead_time
                total_systemoverhead_time += (time_to_completion_worker[0]/cfg.N_PARTITIONS)
                total_computation_time += (time_to_completion_worker[1]/cfg.N_PARTITIONS)

                local_model = deserialize_data(local_model)
                collected_results.append(local_model)
                if len(collected_results) == len(train_partitions):
                    turn.log.info("avg computation per worker at %r is %r", current_epoch, total_computation_time)
                    updated_global_model = mnist.update_model(collected_results,train_partitions)

                    turn.log.info(f"Epoch: {current_epoch} / {cfg.N_EPOCHS} | Final Model Evaluation: {updated_global_model.evaluate(test_set)} / {len(test_set)}")
                    turn.stop(facet = epoch_facet,continuation = lambda: run_epoch(updated_global_model, current_epoch + 1))

            start_read_dist_time = time.perf_counter()
            for part in train_partitions:
                turn.publish(ds, Job(TrainingJob(serialize_data(global_model),serialize_data(part),Embedded(turn.ref(k)))))
            end_read_dist_time = time.perf_counter()
            nonlocal total_systemoverhead_time
            total_systemoverhead_time += (end_read_dist_time - start_read_dist_time)

    ttc_start = time.perf_counter()
    run_epoch(mnist.Network(cfg.NETWORK_ARCHITECTURE), 1)
