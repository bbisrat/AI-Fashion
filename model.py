# Import necessary libraries
import logging
import google.cloud.logging as cloud_logging
from google.cloud.logging.handlers import CloudLoggingHandler
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Setup cloud logging
cloud_logger = logging.getLogger('cloudLogger')
cloud_logger.setLevel(logging.INFO)
cloud_logger.addHandler(CloudLoggingHandler(cloud_logging.Client()))
cloud_logger.addHandler(logging.StreamHandler())

# Load and prepare the Fashion MNIST dataset
(ds_train, ds_test), info = tfds.load('fashion_mnist', split=['train', 'test'], with_info=True, as_supervised=True)
image_batch, labels_batch = next(iter(ds_train))  # Get first batch
print("Before normalization min, max->", np.min(image_batch[0]), np.max(image_batch[0]))

BATCH_SIZE = 32
# Normalize and batch the dataset
normalize = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
ds_train = ds_train.map(normalize).batch(BATCH_SIZE)
ds_test = ds_test.map(normalize).batch(BATCH_SIZE)
image_batch, labels_batch = next(iter(ds_train))  # Check values post-normalization
print("After normalization min, max->", np.min(image_batch[0]), np.max(image_batch[0]))

# Define and compile the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# Train and evaluate the model
model.fit(ds_train, epochs=5)
evaluation = model.evaluate(ds_test)
cloud_logger.info(evaluation)

# Save and reload the model
model.save('saved_model')
new_model = tf.keras.models.load_model('saved_model')
new_model.summary()

# Save and reload model in HDF5 format
model.save('my_model.h5')
new_model_h5 = tf.keras.models.load_model('my_model.h5')
new_model_h5.summary()
