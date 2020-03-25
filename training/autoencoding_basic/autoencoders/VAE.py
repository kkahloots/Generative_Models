import tensorflow as tf
from keras_radam import RAdam
from evaluation.quantitive_metrics.metrics import create_metrics

from graphs.basics.VAE_graph import create_graph, encode_fn, create_losses
from statistical.pdfs import log_normal_pdf
from training.autoencoding_basic.autoencoders.autoencoder import autoencoder

class VAE(autoencoder):
    def __init__(
            self,
            **kwargs
    ):
        kwargs['model_fn'] = create_graph
        kwargs['encode_fn'] = encode_fn
        autoencoder.__init__(
            self,
            **kwargs
        )

    def connect(self, **kwargs):
        # mean, logvariance = self.encode(inputs)
        # z = reparametrize(mean, logvariance)
        # connect the graph x' = decode(z)
        inputs_dict= {
            'x_mean': self.get_variables()['inference_mean'].inputs[0],
            'x_logvariance': self.get_variables()['inference_logvariance'].inputs[0]
        }
        encoded = self.encode(inputs=inputs_dict)
        x_logits = self.decode(encoded['z_latent'])

        logpdf = log_normal_pdf(
            sample=encoded['z_latent'],
            mean=encoded['inference_mean'],
            logvariance=encoded['inference_logvariance']
        )

        # renaming
        x_logits._name = 'x_logits'
        encoded['z_latent']._name = 'z_latent'
        encoded['inference_mean']._name = 'inference_mean'
        encoded['inference_logvariance']._name = 'inference_logvariance'
        outputs_dict = {
            'x_logits': x_logits,
            'z_latent': encoded['z_latent'],
            'x_mean': encoded['inference_mean'],
            'x_logvariance': encoded['inference_logvariance'],
            'logpdf': logpdf
        }

        tf.keras.Model.__init__(
            self,
            name=self.name,
            inputs=inputs_dict,
            outputs=outputs_dict,
            **kwargs
        )

    def outputs_renaming_fn(self):
        ## rename the outputs
        for i, output_name in enumerate(self.output_names):
            if 'logpdf' in output_name:
                self.output_names[i] = 'x_logpdf'
            elif 'z_latent' in output_name:
                self.output_names[i] = 'z_latent'
            elif 'x_logits' in output_name:
                self.output_names[i] = 'x_logits'
            elif 'logvariance' in output_name:
                self.output_names[i] = 'inference_logvariance'
            elif 'inference_mean' in output_name:
                self.output_names[i] = 'inference_mean'
            else:
                pass

    def compile(
            self,
            optimizer=RAdam(),
            loss=None,
            metrics=create_metrics(),
            **kwargs
    ):
        ae_losses = create_losses()
        loss = loss or {}
        for k in loss:
            ae_losses.pop(k)
        self.ae_losses = {**ae_losses, **loss}
        self.ae_metrics = metrics
        tf.keras.Model.compile(self, optimizer=optimizer, loss=self.ae_losses, metrics=self.ae_metrics, **kwargs)
        print(self.summary())

    def batch_cast(self, batch):
        if self.input_kw:
            x = tf.cast(batch[self.input_kw], dtype=tf.float32) / self.input_scale
        else:
            x = tf.cast(batch, dtype=tf.float32) / self.input_scale
        return {
                   'inference_logvariance_inputs': x,
                   'inference_mean_inputs': x
               }, \
               {
                   'x_logits': x,
                   'z_latent': 0.0,
                   'x_logpdf':0.0
               }
