
import tensorflow as tf

from graphs.basics.AE_graph import create_graph, encode_fn
from training.traditional.autoencoders.autoencoder import autoencoder as basicAE

class autoencoder(basicAE):

    def cast_batch(self, xt0, xt1):
        xt0 = tf.cast(xt0, dtype=tf.float32)/self.input_scale
        xt1 = tf.cast(xt1, dtype=tf.float32) / self.input_scale

        return {
                   'inference_inputs': xt0
               }, \
               {
                   'x_logits': xt1
               }
