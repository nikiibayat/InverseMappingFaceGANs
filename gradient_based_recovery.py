import numpy as np
import tensorflow
import tensorflow.compat.v1 as tf
from PIL import Image
import random
import os
import tensorflow_hub as hub
import time
import pickle

tf.disable_eager_execution()

# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

os.environ['TFHUB_CACHE_DIR'] = './'
generator = hub.Module("http://tfhub.dev/google/progan-128/1")
print('Downloaded the model.')


def generate(z_vector):
    return generator(z_vector)


def rescale(batch):
    return batch / 255


def main():
    BATCH_SIZE = 50
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=rescale)
    data_generator = image_datagen.flow_from_directory(
        'generated_test',
        target_size=(128, 128),
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode='sparse')

    index = 0
    z_vectors = []

    while index <= data_generator.batch_index:
        x_batch, y_batch = data_generator.next()
        print("length of batch: ", len(x_batch))
        start_time = time.time()

        fz = tf.Variable(x_batch, tf.float32)
        fz = tf.cast(fz, tf.float32)

        zp = tf.Variable(np.random.normal(size=(len(x_batch), 512)), dtype=tf.float32)
        fzp = generate(zp)

        loss = tf.keras.losses.MeanAbsoluteError()(fz, fzp)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.constant(0.01)
        opt = tf.train.AdamOptimizer(learning_rate)

        train = opt.minimize(loss, var_list=zp, global_step=global_step)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(200):
            print("step: ", i + 1)
            _, loss_value, zp_val, eta = sess.run((train, loss, zp, learning_rate))

            """Comment below lines if you want to remove stochastic clipping"""
            # stochastic clipping
            zp_val[zp_val > 1] = random.uniform(-1, 1)
            zp_val[zp_val < -1] = random.uniform(-1, 1)

        zp_val = sess.run(zp)
        size = zp_val.shape[0]
        if size == BATCH_SIZE:
            z_vectors.append(zp_val)

        print("-" * 50)
        print("Time for one batch:  %s seconds" % (time.time() - start_time))
        print("-" * 50)
        index += 1

        idx = (index - 1) * BATCH_SIZE
        filenames = data_generator.filenames[idx: idx + min(size, BATCH_SIZE)]

        imgs = sess.run(generate(zp))
        imgs = (imgs * 255).astype(np.uint8)

        hr_imgs = sess.run(fz)
        hr_imgs = (hr_imgs * 255).astype(np.uint8)

        dir_name = "stochastic_clipping"
        # dir_name = "gradient_descent"
        if not os.path.exists("results/" + dir_name):
            os.mkdir("results/" + dir_name)

        for i in range(len(filenames)):
            filename = tf.strings.split([filenames[i]], os.path.sep).values[-1]
            filename = tf.strings.split([filename], ".").values[0]
            filename = sess.run(filename).decode('utf-8')
            Image.fromarray(imgs[i]).save("results/{}/{}.png".format(dir_name, filename))


if __name__ == "__main__":
    main()

