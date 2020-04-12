
import tensorflow as tf
from training.autoencoding_basic.autoencoders.VAE import VAE as baseVAE

class VAE(baseVAE):
    def __init__(
            self,
            episode_len,
            **kwargs
    ):
        self.episode_len = episode_len
        baseVAE.__init__(self, **kwargs)


    def get_flat_shape(self):
        return (self.batch_size * self.episode_len, ) + self.get_variables()['generative'].outputs[0].shape[1:][-3:]

    # batch_size =  batch_size * Episode_Len
    def batch_cast(self, xt0, xt1):
        xt0 = tf.cast(xt0, dtype=tf.float32)/self.input_scale
        xt1 = tf.cast(xt1, dtype=tf.float32) / self.input_scale
        return {
                   'inference_logvariance_inputs': xt0,
                   'inference_mean_inputs': xt0
               }, \
               {
                   'x_logits': xt1,
                   'z_latents': 0.0,
                   'x_logpdf':0.0,
                   'x_logvariance': 0.0,
                   'x_mean': 0.0
               }
