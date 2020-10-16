import numpy as np
import tensorflow as tf
from PIL import Image
import time

import os
import tensorflow_hub as hub

# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

os.environ['TFHUB_CACHE_DIR'] = './'
generator = hub.load("http://tfhub.dev/google/progan-128/1").signatures['default']
print('Downloaded the model.')

if not os.path.exists('./generated_test'):
    os.mkdir("./generated_test")
    os.mkdir("./generated_test/test")


def rescale(batch):
    return batch / 255


BATCH_SIZE = 50
np.random.seed(0)

start_time = time.time()

z_vectors = []
for i in range(1):
    print("Batch: ", i)
    zp = tf.convert_to_tensor(np.random.random(size=(BATCH_SIZE, 512)), dtype=tf.float32)
    fzp = generator(zp)['default']

    z_vectors.append(zp.numpy())
    fzp = fzp.numpy()
    fzp = (fzp * 255).astype(np.uint8)

    for j in range(BATCH_SIZE):
        print("face_{} saved.".format(str((i * BATCH_SIZE) + j)))
        Image.fromarray(fzp[j]).save("./generated_test/test/face_{}.png".format(str((i * BATCH_SIZE) + j)))

    print("Time for one batch: ", time.time() - start_time)

# z_reults = np.array(z_vectors).reshape(-1, 512)
# np.save("z_vectors/generated_test_faces", z_reults)
# print("Saved z vectors shape: ", z_reults.shape)
