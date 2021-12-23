import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
from tqdm import tqdm
import emoji
from datetime import datetime



def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(1, input_dim=784))
    model.add(keras.layers.Dense(4))
    model.add(keras.layers.Dense(10))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model

class LayerOutputLogger(keras.callbacks.Callback):
    def __init__(self, validation_ds, log_dir):
        self.validation_ds = validation_ds
        self.writer = tf.summary.create_file_writer(str(log_dir))
        self.dataset_batches = np.sum([1 for i in self.validation_ds])

    def on_epoch_end(self, epoch, logs=None):
        print(emoji.emojize(f':rocket: Epoch {epoch} :rocket: ... calculating outputs for intermediate layers :heart_suit: '))
        with self.writer.as_default():
            for layer_index, layer in enumerate(self.model.layers):
                get_layer_output = K.function(
                    [self.model.layers[0].input],
                    [self.model.layers[layer_index].output],
                )
                for val_data in tqdm(
                    self.validation_ds,
                    total=self.dataset_batches,
                    desc=emoji.emojize(f"Calculating output for layer {layer_index} :rocket: :star-struck:"),
                    unit="batches",
                ):
                    input_data = np.reshape(val_data[0], [-1, 784])
                    layer_output = get_layer_output(input_data)[0]
                    tf.summary.histogram(
                        f"{layer_index}-{str(layer.name)}",
                        layer_output,
                        step=epoch,
                    )


# Load example MNIST data and pre-process it
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # np.shape(x_train) = (60000, 28, 28)

x_train = x_train.reshape(-1, 784).astype("float32") / 255.0   # np.shape(x_train) = (60000, 784)
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Limit the data to 1000 samples
x_train = x_train[:1000] # np.shape(x_train) = (1000, 784)
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]

x_train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))  #transform training data to data.Dataset object. This step can be skipped if the callaback function is updated so a tensor can be passed directly.


current_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
tensorboard_log_dir = f"tensorboard_{current_time}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    histogram_freq=1,
    update_freq="epoch",
    log_dir=tensorboard_log_dir,
)

model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=10,
    verbose=0,
    validation_split=0.5,
    callbacks=[tensorboard_callback, LayerOutputLogger(x_train_dataset, tensorboard_log_dir)],
)

res = model.evaluate(
    x_test, y_test, batch_size=128, verbose=0
)

res = model.predict(x_test, batch_size=128)

