import os
import tensorflow as tf
import numpy as np
from PIL import Image

class trace_reconstruction(tf.keras.callbacks.Callback):
    def __init__(
            self,
            filepath,
            image,
            gen_freq=5,
            **kws
    ):
        self.gen_freq = gen_freq
        self.filepath = filepath
        self.image = image

        tf.keras.callbacks.Callback.__init__(self, **kws)

    def on_train_end(self, logs=None):
        self.reconstruct(-999, logs)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.gen_freq == 0:  # or save after some epoch, each k-th epoch etc.
            self.reconstruct(epoch, logs)

    def reconstruct(self, epoch, logs={}):
        reconstructed = self.model.reconstruct(images=self.image).numpy()
        reconstructed = (reconstructed * 255).astype(np.uint8)[0]
        reconstructed = Image.fromarray(reconstructed, mode='RGB')

        fig_name = os.path.join(self.filepath, 'image_at_epoch_{:06d}.png'.format(epoch))
        reconstructed.save(fig_name)
