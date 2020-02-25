import tensorflow as tf
from graphs.basics.VAE_graph import create_graph, encode_fn, create_losses
from training.traditional.autoencoders.autoencoder import autoencoder
from stats.pdfs import log_normal_pdf

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
        # mean, logvar = self.encode(inputs)
        # z = reparametrize(mean, logvar)
        # connect the graph x' = decode(z)
        _inputs = {
            'x_mean': self.get_variables()['encoder_mean'].inputs[0],
            'x_logvar': self.get_variables()['encoder_logvar'].inputs[0]
        }
        encoded = self.encode(inputs=_inputs)
        x_logits = self.decode(encoded['x_latent'])

        log_pdf = log_normal_pdf(
            sample=encoded['x_latent'],
            mean=encoded['x_mean'],
            logvar=encoded['x_logvar']
        )

        # renaming
        x_logits._name = 'x_logits'
        encoded['x_latent']._name = 'x_latent'
        encoded['x_mean']._name = 'x_mean'
        encoded['x_logvar']._name = 'x_logvar'
        _outputs = {
            'x_logits': x_logits,
            'x_latent': encoded['x_latent'],
            'x_mean': encoded['x_mean'],
            'x_logvar': encoded['x_logvar'],
            'log_pdf': log_pdf
        }

        tf.keras.Model.__init__(
            self,
            name=self.name,
            inputs=_inputs,
            outputs=_outputs,
            **kwargs
        )

    def outputs_renaming_fn(self):
        ## rename the outputs
        for i, _output in enumerate(self.output_names):
            if _output == 'tf_op_layer_log_pdf':
                self.output_names[i] = 'x_log_pdf'
            elif _output == 'tf_op_layer_x_latent':
                self.output_names[i] = 'x_latent'
            elif _output == 'tf_op_layer_x_logits':
                self.output_names[i] = 'x_logits'
            elif _output == 'encoder_logvar':
                self.output_names[i] = 'x_logvar'
            elif _output == 'encoder_mean':
                self.output_names[i] = 'x_mean'
            else:
                pass

    def compile(
            self,
            **kwargs
    ):
        kwargs['loss'] = create_losses()
        autoencoder.compile(self, **kwargs)


    def cast_batch(self, batch):
        if self.input_kw:
            x = tf.cast(batch[self.input_kw], dtype=tf.float32) / self.input_scale
        else:
            x = tf.cast(batch, dtype=tf.float32) / self.input_scale
        return {
                   'encoder_logvar_inputs': x,
                   'encoder_mean_inputs': x
               }, \
               {
                   'x_logits': x,
                   'x_latent': 0.0,
                   'x_log_pdf':0.0,
                   'x_logvar': 0.0,
                   'x_mean': 0.0
               }
