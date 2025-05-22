import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "99"
import json
import math
import tensorflow as tf
import config

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

(mnist_images, mnist_labels), (test_images, test_labels) = (tf.keras.datasets.mnist.load_data())
mnist_images = (mnist_images.astype("float32") / 255.0)[..., None]
test_images  = (test_images.astype("float32")  / 255.0)[..., None]

batch_size    = config.MINI_BATCH_SIZE
LOCAL_EPOCHS  = config.SGD_EPOCHS
ROUNDS        = config.N_EPOCHS

# Prepare datasets
train_dataset = tf.data.Dataset.from_tensor_slices((mnist_images, tf.one_hot(mnist_labels, 10)))
train_dataset = train_dataset.shuffle(60000).batch(batch_size).repeat()
test_dataset  = tf.data.Dataset.from_tensor_slices((test_images, tf.one_hot(test_labels, 10))).batch(batch_size)

cluster_spec         = strategy.cluster_resolver.cluster_spec()
num_workers          = len(cluster_spec.as_dict()['worker'])
samples_per_worker   = mnist_images.shape[0] // num_workers
batches_per_worker   = math.ceil(samples_per_worker / batch_size)
steps_per_round      = batches_per_worker * LOCAL_EPOCHS



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

    task_id = strategy.cluster_resolver.task_id
    callbacks = []
    if task_id == 0:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_prefix,
                save_weights_only=True,
                save_freq='epoch'
            )
        )

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

os._exit(0)