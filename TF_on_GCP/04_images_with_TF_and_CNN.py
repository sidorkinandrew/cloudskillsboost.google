# GSP633
# git clone --depth=1 https://github.com/GoogleCloudPlatform/training-data-analyst
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

# ! gsutil mb -l $REGION $BUCKET_NAME
import os
import sys

from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)

TRAIN_GPU, TRAIN_NGPU = (None, None)
DEPLOY_GPU, DEPLOY_NGPU = (None, None)

TRAIN_VERSION = "tf-cpu.2-8"
DEPLOY_VERSION = "tf2-cpu.2-8"

TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/{}:latest".format(TRAIN_VERSION)
DEPLOY_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/{}:latest".format(DEPLOY_VERSION)

print("Training:", TRAIN_IMAGE, TRAIN_GPU, TRAIN_NGPU)
print("Deployment:", DEPLOY_IMAGE, DEPLOY_GPU, DEPLOY_NGPU)

MACHINE_TYPE = "n1-standard"

VCPU = "4"
TRAIN_COMPUTE = MACHINE_TYPE + "-" + VCPU
print("Train machine type", TRAIN_COMPUTE)

MACHINE_TYPE = "n1-standard"

VCPU = "4"
DEPLOY_COMPUTE = MACHINE_TYPE + "-" + VCPU
print("Deploy machine type", DEPLOY_COMPUTE)

# %%writefile task.py
# Training Fashion MNIST using CNN

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import os
import sys
tfds.disable_progress_bar()

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', dest='epochs',
                    default=10, type=int,
                    help='Number of epochs.')

args = parser.parse_args()

print('Python Version = {}'.format(sys.version))
print('TensorFlow Version = {}'.format(tf.__version__))
print('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
print('DEVICES', device_lib.list_local_devices())

# Define batch size
BATCH_SIZE = 32

# Load the dataset
datasets, info = tfds.load('fashion_mnist', with_info=True, as_supervised=True)

# Normalize and batch process the dataset
ds_train = datasets['train'].map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).batch(BATCH_SIZE)


# Build the Convolutional Neural Network
# As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size.
# Here, the CNN is configured to process inputs of shape (28, 28, 1), which is the format of Fashion MNIST images
# You start with a stack of Conv2D and MaxPooling2D layers.

# Conv2D layer https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
# The number of output channels for each Conv2D layer is controlled by the first argument (e.g., 32 or 64). 
# The next argument specifies the size of the Convolution, in this case, a 3x3 grid.
# `relu` is used as the activation function, which you might recall is the equivalent of returning x when x>0, else returning 0.

# MaxPooling2D layer https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
# By specifying (2,2) for the MaxPooling, the effect is to quarter the size of the image.

# Dense layers
# The output from the convolution stack is flattened and passed to a set of dense layers for classification.
model = tf.keras.models.Sequential([                               
      tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
model.compile(optimizer = tf.keras.optimizers.Adam(),
      loss = tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])



# Train and save the model

MODEL_DIR = os.getenv("AIP_MODEL_DIR")

model.fit(ds_train, epochs=args.epochs)
model.save(MODEL_DIR)

job = aiplatform.CustomTrainingJob(
    display_name=JOB_NAME,
    script_path="task.py",
    container_uri=TRAIN_IMAGE,
    requirements=["tensorflow_datasets==4.6.0"],
    model_serving_container_image_uri=DEPLOY_IMAGE,
)

MODEL_DISPLAY_NAME = "fashionmnist-" + TIMESTAMP

# Start the training
if TRAIN_GPU:
    model = job.run(
        model_display_name=MODEL_DISPLAY_NAME,
        args=CMDARGS,
        replica_count=1,
        machine_type=TRAIN_COMPUTE,
        accelerator_type=TRAIN_GPU.name,
        accelerator_count=TRAIN_NGPU,
    )
else:
    model = job.run(
        model_display_name=MODEL_DISPLAY_NAME,
        args=CMDARGS,
        replica_count=1,
        machine_type=TRAIN_COMPUTE,
        accelerator_count=0,
    )

DEPLOYED_NAME = "fashionmnist_deployed-" + TIMESTAMP

TRAFFIC_SPLIT = {"0": 100}

MIN_NODES = 1
MAX_NODES = 1

#if DEPLOY_GPU:
#    endpoint = model.deploy(
#        deployed_model_display_name=DEPLOYED_NAME,
#        traffic_split=TRAFFIC_SPLIT,
#        machine_type=DEPLOY_COMPUTE,
#        accelerator_type=DEPLOY_GPU.name,
#        accelerator_count=DEPLOY_NGPU,
#        min_replica_count=MIN_NODES,
#        max_replica_count=MAX_NODES,
#    )


endpoint = model.deploy(
        deployed_model_display_name=DEPLOYED_NAME,
        traffic_split=TRAFFIC_SPLIT,
        machine_type=DEPLOY_COMPUTE,
        accelerator_type=None,
        accelerator_count=0,
        min_replica_count=MIN_NODES,
        max_replica_count=MAX_NODES,
    )

print(TIMESTAMP)



import tensorflow_datasets as tfds
import numpy as np

tfds.disable_progress_bar()

datasets, info = tfds.load('fashion_mnist', batch_size=-1, with_info=True, as_supervised=True)

test_dataset = datasets['test']

x_test, y_test = tfds.as_numpy(test_dataset)

# Normalize (rescale) the pixel data by dividing each pixel by 255. 
x_test = x_test.astype('float32') / 255.

x_test.shape, y_test.shape

#@title Pick the number of test images
NUM_TEST_IMAGES = 20 #@param {type:"slider", min:1, max:20, step:1}
x_test, y_test = x_test[:NUM_TEST_IMAGES], y_test[:NUM_TEST_IMAGES]

predictions = endpoint.predict(instances=x_test.tolist())
y_predicted = np.argmax(predictions.predictions, axis=1)

correct = sum(y_predicted == np.array(y_test.tolist()))
total = len(y_predicted)
print(
    f"Correct predictions = {correct}, Total predictions = {total}, Accuracy = {correct/total}"
)
