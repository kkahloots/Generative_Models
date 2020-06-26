import tensorflow as tf
from keras_radam import RAdam
from evaluation.quantitive_metrics.metrics import create_metrics

from graphs.basics.VAE_graph import create_graph, encode_fn, create_tlosses
from statistical.pdfs import log_Student_df1_pdf, log_Student_df05_pdf
from training.autoencoding_basic.autoencoders.autoencoder import autoencoder

class tVAE(autoencoder):
    def __init__(
            self,
            df=1,
            **kwargs
    ):
        self.df=df
        kwargs['model_fn'] = create_graph
        kwargs['encode_fn'] = encode_fn
        autoencoder.__init__(
            self,
            **kwargs
        )

    # autoencoder function
    def encode(self, x):
        return self.__encode__(inputs={'x_mean': x, 'x_logvariance': x})['z_latents']

    def __init_autoencoder__(self, **kwargs):
        # mean, logvariance = self.__encode__(inputs)
        # z = reparametrize(mean, logvariance)
        # connect the graph x' = decode(z)
        inputs_dict= {
            'x_mean': self.get_variables()['inference_mean'].inputs[0],
            'x_logvariance': self.get_variables()['inference_logvariance'].inputs[0]
        }
        encoded = self.__encode__(inputs=inputs_dict)
        x_logits = self.decode(encoded['z_latents'])

        if self.df==0.5:
            logpdf = log_Student_df05_pdf(
                sample=encoded['z_latents'],
                mean=encoded['inference_mean'],
                logvariance=encoded['inference_logvariance']
            )
        else:
            logpdf = log_Student_df1_pdf(
                sample=encoded['z_latents'],
                mean=encoded['inference_mean'],
                logvariance=encoded['inference_logvariance']
            )

        # renaming
        x_logits._name = 'x_logits'
        encoded['z_latents']._name = 'z_latents'
        encoded['inference_mean']._name = 'inference_mean'
        encoded['inference_logvariance']._name = 'inference_logvariance'
        outputs_dict = {
            'x_logits': x_logits,
            'z_latents': encoded['z_latents'],
            'x_mean': encoded['inference_mean'],
            'x_logvariance': encoded['inference_logvariance'],
            'x_logpdf': logpdf
        }

        tf.keras.Model.__init__(
            self,
            name=self.name,
            inputs=inputs_dict,
            outputs=outputs_dict,
            **kwargs
        )

    def __rename_outputs__(self):
        ## rename the outputs
        for i, output_name in enumerate(self.output_names):
            if 'logpdf' in output_name:
                self.output_names[i] = 'x_logpdf'
            elif 'z_latents' in output_name:
                self.output_names[i] = 'z_latents'
            elif 'x_logits' in output_name:
                self.output_names[i] = 'x_logits'
            elif 'logvariance' in output_name:
                self.output_names[i] = 'inference_logvariance'
            elif 'inference_mean' in output_name:
                self.output_names[i] = 'inference_mean'
            else:
                pass

    # override function
    def compile(
            self,
            optimizer=RAdam(),
            loss=None,
            **kwargs
    ):
        ae_losses = create_tlosses(log_Student_df05_pdf) if self.df==0.5 else create_tlosses(log_Student_df1_pdf)
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

    def get_inputs_shape(self):
        return list(self.get_variables()['inference_mean'].inputs[0].shape[1:])

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
                   'z_latents': 0.0,
                   'x_logpdf':0.0
               }
