
import tensorflow as tf
from training.autoencoding_basic.autoencoders.autoencoder import autoencoder as basicAE

class autoencoder(basicAE):
    def __init__(
            self,
            episode_len,
            **kwargs
    ):
        self.episode_len = episode_len
        basicAE.__init__(self, **kwargs)

    def get_flat_shape(self):
        return (self.batch_size * self.episode_len, ) + self.get_variables()['generative'].outputs[0].shape[1:][-3:]

    def batch_cast(self, xt0, xt1):

        return {
                   'inference_inputs': xt0
               }, \
               {
                   'x_logits': xt1
               }
