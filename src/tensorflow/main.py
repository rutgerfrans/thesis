import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "99"
import json, math, time, csv, random
import tensorflow as tf
import config
import pandas as pd

fault_p = config.FAULT_P

class FaultInjectionCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        # each batch has a chance fault_p to kill this process immediately
        if fault_p > 0.0 and random.random() < fault_p:
            print(f"[worker {task_id}] Injecting fault at batch {batch}")
            os._exit(1)

class EpochTimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        # prepare in-memory accumulators
        self.epoch_batch_time = 0.0
        self.epoch_read_time  = 0.0
        self.epoch_comm_time  = 0.0
        self.last_batch_end   = time.perf_counter()
        self.batch_in_epoch   = 0

        # ensure output CSV exists with header
        self.csv_path = os.path.join(os.getcwd(), "src/tensorflow/timings/epoch_timings.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "worker", "epoch",
                    "total_batch_time",
                    "total_read_time",
                    "total_comm_time"
                ])

    def on_train_batch_begin(self, batch, logs=None):
        # mark the moment we start “reading” next batch
        now = time.perf_counter()
        self.read_start = now

    def on_train_batch_end(self, batch, logs=None):
        end = time.perf_counter()
        # total time this batch (including any idle from last_batch_end)
        batch_time = end - self.last_batch_end
        # read time = from last batch end until batch_begin
        read_time  = self.read_start - self.last_batch_end
        # comm+compute = rest
        comm_time  = batch_time - read_time

        # accumulate
        self.epoch_batch_time += batch_time
        self.epoch_read_time  += read_time
        self.epoch_comm_time  += comm_time
        self.batch_in_epoch   += 1
        self.last_batch_end    = end

    def on_epoch_end(self, epoch, logs=None):
        # figure out worker id
        resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        worker_id = resolver.task_id

        # append one line for this epoch
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                worker_id,
                epoch + 1,  # 1-based epoch count
                f"{self.epoch_batch_time:.6f}",
                f"{self.epoch_read_time:.6f}",
                f"{self.epoch_comm_time:.6f}"
            ])

        # reset accumulators for next epoch
        self.epoch_batch_time = 0.0
        self.epoch_read_time  = 0.0
        self.epoch_comm_time  = 0.0
        self.last_batch_end   = time.perf_counter()
        self.batch_in_epoch   = 0

tf.get_logger().setLevel('ERROR')
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if "TF_CONFIG" not in os.environ:
    tf_config = {
        "cluster": {"worker": ["localhost:12345"]},
        "task":    {"type": "worker", "index": 0}
    }
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
print("TF_CONFIG:", os.environ["TF_CONFIG"])

strategy = tf.distribute.MultiWorkerMirroredStrategy()
initial_epoch = 0

(mnist_images, mnist_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
mnist_images = (mnist_images.astype("float32") / 255.0)[..., None]
test_images  = (test_images.astype("float32")  / 255.0)[..., None]

batch_size    = config.MINI_BATCH_SIZE
LOCAL_EPOCHS  = config.SGD_EPOCHS
ROUNDS        = config.N_EPOCHS

# Prepare datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
    (mnist_images, tf.one_hot(mnist_labels, 10))
).shuffle(60000).batch(batch_size).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_images, tf.one_hot(test_labels, 10))
).batch(batch_size)

cluster_spec       = strategy.cluster_resolver.cluster_spec()
num_workers        = len(cluster_spec.as_dict()['worker'])
samples_per_worker = mnist_images.shape[0] // num_workers
batches_per_worker = math.ceil(samples_per_worker / batch_size)
steps_per_round    = batches_per_worker * LOCAL_EPOCHS

cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
task_type, task_id = cluster_resolver.task_type, cluster_resolver.task_id
is_chief = (task_type == "worker" and task_id == 0)

if is_chief:
    print(f"num_workers={num_workers}  samples_per_worker={samples_per_worker}  "
          f"batches_per_worker={batches_per_worker}  steps_per_round={steps_per_round}")

with strategy.scope():
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,1)),
        tf.keras.layers.Dense(16, activation="sigmoid"),
        tf.keras.layers.Dense(16, activation="sigmoid"),
        tf.keras.layers.Dense(10, activation="sigmoid"),
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.ETA)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["accuracy"]
    )

    checkpoint_dir = os.path.join(os.getcwd(), 'src','tensorflow','checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt-{epoch}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        if is_chief:
            print(f"Loading weights from {latest_ckpt}")
        model.load_weights(latest_ckpt)
        try:
            initial_epoch = int(latest_ckpt.split('-')[-1])
        except ValueError:
            initial_epoch = 0

    # Build callbacks list
    callbacks = []
    if is_chief:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_prefix,
                save_weights_only=True,
                save_freq='epoch'
            )
        )
    # append timing callback on every worker
    callbacks.append(EpochTimingCallback())
    callbacks.append(FaultInjectionCallback())

# Train (all workers)
model.fit(
    train_dataset,
    epochs=ROUNDS,
    initial_epoch=initial_epoch,
    steps_per_epoch=steps_per_round,
    validation_data=test_dataset,
    validation_steps=math.ceil(test_images.shape[0] / batch_size),
    verbose=1 if is_chief else 0,
    callbacks=callbacks
)

# after model.fit(…) and before os._exit(0):
if is_chief:
    # read the single per‐epoch log
    df = pd.read_csv("epoch_timings.csv")

    # sum up per worker
    per_worker = df.groupby("worker").sum()[[
        "total_batch_time", "total_read_time", "total_comm_time"
    ]]

    print("\nPer-worker total timings (seconds):")
    for worker, row in per_worker.iterrows():
        print(f"  worker {int(worker):>2}:  "
              f"batch={row.total_batch_time:.2f}, "
              f"read={row.total_read_time:.2f}, "
              f"comm+comp={row.total_comm_time:.2f}")

    # compute averages
    avg = per_worker.mean()
    print(f"\nAverage over {len(per_worker)} workers:  "
          f"batch={avg.total_batch_time:.2f}, "
          f"read={avg.total_read_time:.2f}, "
          f"comm+comp={avg.total_comm_time:.2f}\n")

os._exit(0)
