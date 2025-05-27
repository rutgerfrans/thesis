import os, glob, math, time, csv, json, random
from datetime import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_VLOG_LEVEL"] = "99"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import config
import tensorflow as tf
from tensorflow.keras.callbacks import BackupAndRestore
from src.data_loader import load_dataset
import numpy as np

fault_p = config.FAULT_P

def clear_snapshots(checkpoint_dir):
    for fname in glob.glob(os.path.join(checkpoint_dir, 'ckpt-*')):
        try:
            os.remove(fname)
            print(f"Deleted checkpoint: {fname}")
        except OSError as e:
            print(f"Error deleting {fname}: {e}")

class FaultInjectionCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, batch, logs=None):
        if fault_p > 0.0 and random.random() < fault_p:
            resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
            print(f"[worker {resolver.task_id}] Injecting fault at batch {batch}")
            os._exit(1)

class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.compute_time_total = 0.0
        self.csv_path = os.path.join(os.getcwd(), "src/tensorflow/timings/epoch_timings.csv")
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["system_overhead","avg_compute_time_per_worker"])

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_comp_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        t1 = time.perf_counter()
        comp = t1 - self.epoch_comp_start
        self.compute_time_total += comp
        resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        if resolver.task_type == 'worker' and resolver.task_id == 0:
            print(f"[Epoch {epoch+1}] compute_time={comp:.4f}s")

    def on_train_end(self, logs=None):
        run_start = getattr(self, 'run_start', None)
        run_end = datetime.now()
        total_wall = (run_end - run_start).total_seconds()

        comp_tensor = tf.constant(self.compute_time_total, dtype=tf.float64)
        comp_sum = tf.raw_ops.CollectiveReduce(
            input=comp_tensor,
            group_size=tf.distribute.get_strategy().num_replicas_in_sync,
            group_key=1,
            instance_key=1,
            merge_op='Add',
            final_op='Id',
            subdiv_offsets=[0]
        )
        avg_compute = comp_sum.numpy() / tf.distribute.get_strategy().num_replicas_in_sync

        system_overhead = total_wall - avg_compute

        resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        if resolver.task_type == 'worker' and resolver.task_id == 0:
            print(f"Completed training at {run_end.isoformat()}")
            print(f"avg_compute_per_worker={avg_compute:.4f}s; system_overhead={system_overhead:.4f}s")
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{system_overhead:.6f}", f"{avg_compute:.6f}"])

if __name__ == '__main__':
    run_start = datetime.now()
    TimingCallback.run_start = run_start

    if 'TF_CONFIG' not in os.environ:
        tf_config = {
            'cluster':{'worker':['localhost:12345']},
            'task':  {'type':'worker','index':0}
        }
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    task_type, task_id = resolver.task_type, resolver.task_id
    is_chief = (task_type=='worker' and task_id==0)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print(f"Starting training at {run_start.isoformat()}, task {task_id}")

    partitions, test_data = load_dataset()
    local_data = partitions[task_id]
    x_local = np.stack([img.flatten() for img, _ in local_data])
    x_local = x_local.reshape(-1, 28, 28, 1).astype('float32')
    y_local = np.stack([label.flatten() for _, label in local_data]).astype('float32')
    if config.TRAIN_SAMPLE_SIZE > 0:
        x_local = x_local[:config.TRAIN_SAMPLE_SIZE]
        y_local = y_local[:config.TRAIN_SAMPLE_SIZE]

    batch_size   = config.MINI_BATCH_SIZE
    LOCAL_EPOCHS = config.SGD_EPOCHS
    ROUNDS       = config.N_EPOCHS
    steps_per_epoch = math.ceil(x_local.shape[0] / batch_size) * LOCAL_EPOCHS

    train_dataset = (tf.data.Dataset.from_tensor_slices((x_local, y_local))
                     .shuffle(x_local.shape[0])
                     .batch(batch_size)
                     .repeat())
    
    x_test, y_test_int = zip(*test_data)
    x_test = np.stack([img.flatten() for img in x_test]).reshape(-1, 28, 28, 1).astype('float32')
    y_test = tf.one_hot(np.array(y_test_int), config.NETWORK_ARCHITECTURE[-1])
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    with strategy.scope():
        layers = [tf.keras.layers.Flatten(input_shape=(28,28,1))]
        for units in config.NETWORK_ARCHITECTURE[1:-1]:
            layers.append(tf.keras.layers.Dense(units, activation='sigmoid'))
        layers.append(tf.keras.layers.Dense(config.NETWORK_ARCHITECTURE[-1], activation='sigmoid'))
        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=config.ETA),
            loss='mse',
            metrics=['accuracy']
        )

    # Checkpoint & backup dirs
    ckpt_dir   = os.path.join(os.getcwd(),'src','tensorflow','checkpoints')
    backup_dir = os.path.join(os.getcwd(),'src','tensorflow','backup')
    os.makedirs(ckpt_dir,   exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    ckpt_prefix = os.path.join(ckpt_dir,'ckpt-{epoch}')

    callbacks = [BackupAndRestore(backup_dir), FaultInjectionCallback(), TimingCallback()]
    if is_chief:
        callbacks.insert(0, tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_prefix,
            save_weights_only=True,
            save_freq='epoch'
        ))

    # Launch distributed training
    model.fit(
        train_dataset,
        epochs=ROUNDS,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        verbose=1 if is_chief else 0,
        callbacks=callbacks
    )

    if is_chief:
        clear_snapshots(ckpt_dir)

    os._exit(0)
