from syndicate import relay, Record, patterns as P, dataspace, turn, Embedded
from syndicate.during import During
from src.data_loader import load_dataset
from src.comms.sam_comm import SAMFederatedCommunicator
import src.utils as utils
import config as cfg
import src.mnist as mnist

TrainingJob = Record.makeConstructor('TJob', 'gm tp k') # global_model, training_partition

NUM_EPOCHS = cfg.N_EPOCHS

available_workers = []     # list of worker refs / this is basicly a FIFO queue
jobs_table = []      # list of (FibRequest, worker ref)
waiting_jobs = []      # list of FibRequest
local_models = []
current_epoch = 1 

@relay.service(name='jobqmanager')
@During().add_handler
def main(data):
    global_model_file = "tmp/global_model.pkl"
    comm = SAMFederatedCommunicator()
    train_partitions, test_set = load_dataset()
    global_model = mnist.serialize_network(comm.init_global_model())

    worker_ds  = data['worker-dataspace'].embeddedValue
    #turn.log.info("jobqmanager up")

    def cleanup_worker_exit(worker):
        #turn.log.info("worker-withdrawn %r", worker)
        if worker in available_workers:
            available_workers.remove(worker)

        for job, wk in list(jobs_table):
            if wk == worker:
                turn.log.warning("requeueing job (worker died)")
                waiting_jobs.append(job)
                jobs_table.remove((job, wk))

        dispatch()

    @dataspace.during(worker_ds, P.rec('worker-ready', P.CAPTURE))
    def new_worker(w):
        w = w.embeddedValue
        #turn.log.info("worker-ready %r", w)
        available_workers.append(w)
        dispatch()
        turn.on_stop(lambda w=w: cleanup_worker_exit(w))

    @During().add_handler
    def on_result_tj(local_model, partition):
        local_model = mnist.deserialize_network(local_model)
        local_models.append(local_model)
        #turn.log.info("Client got reply")

        # update jobstable manually, can we do this automatically?
        for job, wk in list(jobs_table):
            if TrainingJob._tp(job) == partition:
                #turn.log.info("freeing worker %r (job tp=%r)", wk, partition)
                available_workers.append(wk)
                jobs_table.remove((job, wk))
                break

        dispatch()

    def create_jobs():
        waiting_jobs.clear()
        data_stack = comm.create_data_stack(global_model_file, train_partitions)
        for stack in data_stack:
            tj = TrainingJob(global_model, stack[0], Embedded(turn.ref(on_result_tj)))
            waiting_jobs.append(tj)

    # Dispatch() tries to dispatch a job and a worker when a pair can be made.
    # It also checks epoch completion
    def dispatch():
        global current_epoch
        while available_workers and waiting_jobs:
            job = waiting_jobs.pop(0)
            wk  = available_workers.pop(0)
            #turn.log.info("dispatching job")
            jobs_table.append((job, wk))
            turn.publish(wk, job)

        if not waiting_jobs and not jobs_table:
            nonlocal global_model
            updated_global_model = comm.update_model(local_models, train_partitions, test_set, current_epoch)
            turn.log.info(f"Epoch: {current_epoch} / {cfg.N_EPOCHS} | Final Model Evaluation: {updated_global_model.evaluate(test_set)} / {len(test_set)}")
            global_model = mnist.serialize_network(updated_global_model)
            if current_epoch < NUM_EPOCHS:
                local_models.clear()
                #turn.log.info("---- epoch %d complete; starting next ----", current_epoch)
                current_epoch += 1
                create_jobs()
                dispatch()
            else:
                turn.log.info("All %d epochs finished!", NUM_EPOCHS)
    
    create_jobs()
    dispatch()
