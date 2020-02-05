import os

import tensorflow as tf
from keras_radam import RAdam

from evaluation.quantitive_metrics.metrics import create_metrics
from graphs.basics.AE_graph import create_graph, create_losses, encode_fn, decode_fn, generate_sample
from graphs.builder import load_models, save_models


class autoencoder(tf.keras.Model):
    def __init__(
            self,
            name,
            inputs_shape,
            outputs_shape,
            latent_dim,
            variables_params,
            filepath=None,
            model_fn=create_graph,
            **kwargs

    ):
        self.get_variables = model_fn(
            name=name,
            variables_params=variables_params,
            restore=filepath
        )
        self.inputs_shape = inputs_shape
        self.outputs_shape = outputs_shape
        self.latent_dim = latent_dim
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.generate_sample = generate_sample
        self.save_models = save_models
        self.load_models = load_models

        # connect the graph x' = decode(encode(x))
        _inputs = {k: v.inputs[0] for k, v in self.get_variables().items() if k=='inference'}
        latent = self.encode(_inputs['inference'])
        x_logits = self.decode(latent)
        _outputs = {'x_logits': x_logits}

        tf.keras.Model.__init__(
            self,
            name=name,
            inputs=_inputs,
            outputs=_outputs,
            **kwargs
        )

        ## rename the outputs
        layer = [layer for layer in self.layers if layer._name.endswith('x_logits')][0]
        layer._name = 'x_logits'
        self.output_names = ['x_logits']


    def compile(self,
              optimizer=RAdam(),
              loss=create_losses(),
              metrics=create_metrics(),
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              distribute=None,
              **kwargs):

        tf.keras.Model.compile(self, optimizer=optimizer, loss=loss, metrics=metrics)
        print(self.summary())

    def encode(self, inputs):
        if inputs.shape == self.inputs_shape:
            inputs = tf.reshape(inputs, (1, ) + self.inputs_shape)
        inputs = tf.cast(inputs, tf.float32)
        return self.encode_fn(model=self.get_varibale, inputs=inputs)

    def decode(self, latent):
        return self.decode_fn(model=self.get_varibale, latent=latent, inputs_shape=self.inputs_shape)

    def get_varibale(self, var_name, param):
        return self.get_variables()[var_name](*param)

    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None):
        file_Name = os.path.join(filepath, self.name)
        self.save_models(file_Name, self.get_variables())

    def feedforwad(self, inputs):
        z = self.encode_fn(model=self.get_varibale, inputs=inputs)
        x_logit = self.decode_fn(model=self.get_varibale, latent=z, inputs_shape=self.inputs_shape)
        return {'x_logit': x_logit, 'latent': z}
