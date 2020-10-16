import os
import tensorflow_hub as hub
import tensorflow as tf
import cv2
import time
import pickle
import numpy as np
import keras
from classification_models.keras import Classifiers

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
PATH_TEST = "generated_faces"
# PATH_TEST = "AR_faces"


BUFFER_SIZE = 1000
BATCH_SIZE = 50
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_LR_WIDTH = 32
IMG_LR_HEIGHT = 32


def load(image_file):
    input_image = tf.io.read_file(image_file)
    input_image = tf.image.decode_jpeg(input_image)
    input_image = tf.cast(input_image, tf.float32)

    return input_image


def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.BICUBIC)

    return input_image


# normalizing the images to [0, 1]
def normalize(input_image):
    return input_image / 255.


def load_image_test(image_file):
    filename = tf.strings.split([image_file], os.path.sep).values[-1]
    filename = tf.strings.split([filename], ".").values[0]
    index = tf.strings.split([filename], "_").values[-1]

    input_image = load(image_file)
    input_image = resize(input_image, IMG_LR_HEIGHT, IMG_LR_WIDTH)
    input_image = resize(input_image, IMG_HEIGHT, IMG_WIDTH)
    input_image = normalize(input_image)

    return input_image, index


def load_hr_image(image_file):
    input_image = load(image_file)
    input_image = resize(input_image, IMG_HEIGHT, IMG_WIDTH)
    input_image = normalize(input_image)

    return input_image


def ResNet():
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dense(2048)(x)
    x = keras.layers.Dense(1024)(x)
    z_vector = keras.layers.Dense(512)(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[z_vector])
    return model


os.environ['TFHUB_CACHE_DIR'] = './'
generator = hub.load("http://tfhub.dev/google/progan-128/1").signatures['default']

"""Building the model"""
ResNet18, preprocess_input = Classifiers.get('resnet18')
base_model = ResNet18(input_shape=(224, 224, 3), weights=None, include_top=False)
resnet_model = ResNet()


optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

""" Uncomment below if you want to test your checkpoints"""
# checkpoint_dir = './checkpoints/training_checkpoints4/'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(optimizer=optimizer,
#                                  Resnet=resnet_model)
# # restoring the latest checkpoint in checkpoint_dir
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

resnet_model = tf.keras.models.load_model('saved_model/resnet_model_chkp25')

z_vectors = np.load('z_vectors/generated_test_faces.npy')
print("Total number of z vectors: ", z_vectors.shape)

print("Tensorflow: ", tf.__version__, " Eager: ", tf.executing_eagerly())
test_dataset = tf.data.Dataset.list_files(PATH_TEST + '/*.png', shuffle=False)
test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)

if not os.path.exists("results"):
    os.mkdir("results")

if not os.path.exists("results/resnet"):
    os.mkdir("results/resnet")

start_time = time.time()
for lr_face, z_indices in test_dataset.take(1):
    z_indices = z_indices.numpy()
    z_indices = [int(z_indices[i].decode("utf-8")) for i in range(len(z_indices))]
    sr_z = resnet_model(tf.keras.backend.expand_dims(lr_face, axis=-1))
    sr_face = generator(sr_z)['default']

    sr_face = np.array(sr_face)
    sr_face = sr_face * 255

    lr_image = np.array(lr_face)
    lr_image = lr_image * 255

    for i in range(len(z_indices)):
        cv2.imwrite("results/resnet/face_{}.png".format(str(z_indices[i])), cv2.cvtColor(sr_face[i], cv2.COLOR_BGR2RGB))

print("Total time for a batch of 50 faces: ", time.time() - start_time)
