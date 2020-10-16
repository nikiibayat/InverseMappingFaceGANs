import tensorflow as tf
import os
import time
import datetime
import tensorflow_hub as hub
import numpy as np
from random import sample
import keras
from classification_models.keras import Classifiers

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)


PATH_TRAIN_generated = "/imaging/nbayat/progan_faces/generated_faces"
PATH_TRAIN_real = "/imaging/nbayat/vggface2/train_100_cropped"
log_dir = "logs/logs6"

BUFFER_SIZE = 1000
BATCH_SIZE = 4
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_LR_WIDTH = 32
IMG_LR_HEIGHT = 32
EPOCHS = 100


def load(image_file):
    input_image = tf.io.read_file(image_file)
    input_image = tf.io.decode_png(input_image)
    input_image = tf.cast(input_image, tf.float32)

    return input_image


def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.BICUBIC)

    return input_image


def normalize(input_image):
    return input_image / 255.


def load_image_train_real(image_file):
    input_image = load(image_file)
    HR_image = normalize(input_image)

    LR_image = resize(input_image, IMG_HEIGHT, IMG_WIDTH)
    LR_image = normalize(LR_image)

    return LR_image, HR_image


def load_image_train_generated(image_file):
    filename = tf.strings.split([image_file], os.path.sep).values[-1]
    filename = tf.strings.split([filename], ".").values[0]
    index = tf.strings.split([filename], "_").values[-1]

    input_image = load(image_file)
    LR_image = resize(input_image, IMG_HEIGHT, IMG_WIDTH)
    LR_image = normalize(LR_image)

    return LR_image, index


def resnet():
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dense(2048)(x)
    x = keras.layers.Dense(1024)(x)
    z_vector = keras.layers.Dense(512)(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[z_vector])
    return model


def compute_perceptual_loss(sr_face, hr_face):
    sr = resize(sr_face, 160, 160)
    hr = resize(hr_face, 160, 160)

    sr_embd = facenet(sr)
    hr_embd = facenet(hr)
    embedding_loss = mae(sr_embd, hr_embd)

    total_loss = 0
    for layer_index in perceptual_layers:
        layer = facenet.layers[layer_index]
        model = tf.keras.Model(facenet.input, layer.output)
        sr_features = model(sr)
        hr_features = model(hr)
        loss = mse(sr_features, hr_features)
        total_loss += loss

    total_loss = total_loss / len(perceptual_layers)
    return total_loss, embedding_loss


@tf.function
def train_step(lr_face, hr_face, epoch, phase2=False):
    with tf.GradientTape() as gen_tape:
        sr_z = resnet_model(preprocess_input(lr_face))
        if phase2:
            sr_face = generator(sr_z)['default']
            pixel_loss = mae(sr_face, hr_face)
            perceptual_loss, embedding_loss = compute_perceptual_loss(sr_face, hr_face)

            loss = pixel_loss + perceptual_loss
        else:
            pixel_loss = mae(sr_z, hr_face)
            hr_face = generator(hr_face)['default']
            sr_face = generator(sr_z)['default']
            perceptual_loss, embedding_loss = compute_perceptual_loss(sr_face, hr_face)
            loss = pixel_loss + perceptual_loss

        resnet_gradients = gen_tape.gradient(loss,
                                             resnet_model.trainable_variables)
        optimizer.apply_gradients(zip(resnet_gradients,
                                      resnet_model.trainable_variables))

        with summary_writer.as_default():
            if not phase2:
                tf.summary.scalar('Pixel loss of generated face z vectors', pixel_loss, step=epoch)
                tf.summary.scalar('Perceptual loss between generated faces', perceptual_loss, step=epoch)
                tf.summary.scalar('Total loss of generated faces', loss, step=epoch)
            else:
                tf.summary.scalar('Pixel loss of real face and reconstructed', pixel_loss, step=epoch)
                tf.summary.scalar('Perceptual loss of real and reconstructed', perceptual_loss, step=epoch)
                tf.summary.scalar('Embedding loss of real face and reconstructed', embedding_loss, step=epoch)
                tf.summary.scalar('Total loss of real faces', loss, step=epoch)
            for weights, grads in zip(resnet_model.trainable_weights, resnet_gradients):
                tf.summary.histogram(weights.name.replace(':', '_') + '_grads', data=grads, step=epoch)


def fit(train_ds_real, train_ds_generated, epochs):
    for epoch in range(epochs):
        start = time.time()
        """Real faces"""
        # for b, (lr_face, hr_face) in enumerate(train_ds_real):
        #     print("Phase2 - Epoch: {} Batch: {}".format(epoch + 1, b + 1))
        #     train_step(lr_face, hr_face, epoch, phase2=True)
        """Generated faces"""
        for b, (lr_face, z_indices) in enumerate(train_ds_generated):
            print("Phase1 - Epoch: {} Batch: {}".format(epoch + 1, b + 1))
            z_indices = z_indices.numpy()
            z_indices = [int(z_indices[i].decode("utf-8")) for i in range(len(z_indices))]
            hr_z_vector = z_vectors[z_indices]
            train_step(lr_face, hr_z_vector, epoch)

        checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))


"""Input Pipeline"""
z_vectors = np.load('z_vectors/generated_face_100k.npy')
print("Total number of z vectors: ", z_vectors.shape)

print("Tensorflow: ", tf.__version__, " Eager: ", tf.executing_eagerly())
train_dataset_real = tf.data.Dataset.list_files(PATH_TRAIN_real + '/*/*.png')
train_dataset_real = train_dataset_real.map(load_image_train_real, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset_real = train_dataset_real.batch(BATCH_SIZE)

train_dataset_generated = tf.data.Dataset.list_files(PATH_TRAIN_generated + '/*.png')
train_dataset_generated = train_dataset_generated.map(load_image_train_generated,
                                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset_generated = train_dataset_generated.batch(16)

"""Building the model"""
os.environ['TFHUB_CACHE_DIR'] = './'
hub_layer = hub.KerasLayer("http://tfhub.dev/google/progan-128/1")
generator = hub.load("http://tfhub.dev/google/progan-128/1").signatures['default']
facenet = tf.keras.models.load_model('facenet.h5')

ResNet34, preprocess_input = Classifiers.get('resnet34')
base_model = ResNet34(input_shape=(224, 224, 3), weights=None, include_top=False)
resnet_model = resnet()

perceptual_layers = []
for i in range(len(facenet.layers)):
  layer_type = facenet.layers[i].name.split("_")[-1]
  if layer_type == "Concatenate":
    perceptual_layers.append(i)
print("Total of {} activation layers found!".format(len(perceptual_layers)))

"""Define Loss and Optimizer"""
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
kl = tf.keras.losses.KLDivergence()
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './checkpoints/training_checkpoints6'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 Resnet=resnet_model)

summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""Train the Model"""
fit(train_dataset_real, train_dataset_generated, EPOCHS)

