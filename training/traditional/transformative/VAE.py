
import tensorflow as tf
from training.traditional.autoencoders.VAE import VAE


class VAE(VAE):
    def cast_batch(self, xt0, xt1):
        xt0 = tf.cast(xt0, dtype=tf.float32)/self.input_scale
        xt1 = tf.cast(xt1, dtype=tf.float32) / self.input_scale
        return {
                   'encoder_logvar_inputs': xt0,
                   'encoder_mean_inputs': xt0
               }, \
               {
                   'x_logits': xt1,
                   'x_latent': 0.0,
                   'x_log_pdf':0.0,
                   'x_logvar': 0.0,
                   'x_mean': 0.0
               }
