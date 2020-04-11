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
            angel=-270,
            **kws
    ):
        self.gen_freq = gen_freq
        self.filepath = filepath
        self.angel = angel
        self.image = image
        original_image = (self.image * 255).astype(np.uint8)
        original_image = Image.fromarray(original_image, mode='RGB')

        fig_name = os.path.join(self.filepath, 'input_image.png')
        original_image.rotate(self.angel, expand=True).save(fig_name)

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
        reconstructed.rotate(self.angel, expand=True).save(fig_name)
