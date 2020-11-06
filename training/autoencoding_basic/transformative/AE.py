
import tensorflow as tf
from tensorflow_addons.optimizers import RectifiedAdam

from evaluation.quantitive_metrics.metrics import create_metrics
from graphs.basics.AE_graph import create_trans_losses
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

    @tf.function
    def batch_cast(self, batch):

        return {
                   'inference_inputs': batch['xt0']
               }, \
               {
                   'x_logits': batch['xt1']
               }
