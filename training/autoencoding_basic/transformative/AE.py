
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

    # override function
    def compile(
            self,
            optimizer=RectifiedAdam(),
            loss=None,
            input_fn=None,
            output_fn=None,
            **kwargs
    ):

        ae_losses = create_trans_losses(input_fn, output_fn)
        loss = loss or {}
        for k in loss:
            ae_losses.pop(k)
        self.ae_losses = {**ae_losses, **loss}

        if 'metrics' in kwargs.keys():
            self.ae_metrics = kwargs.pop('metrics', None)
        else:
            self.ae_metrics = create_metrics(self.get_flat_shape())

        tf.keras.Model.compile(self, optimizer=optimizer, loss=self.ae_losses, metrics=self.ae_metrics, **kwargs)
        print(self.summary())

    def get_flat_shape(self):
        return (self.batch_size * self.episode_len, ) + self.get_variables()['generative'].outputs[0].shape[1:][-3:]

    def batch_cast(self, xt0, xt1):

        return {
                   'inference_inputs': xt0
               }, \
               {
                   'x_logits': xt1
               }
