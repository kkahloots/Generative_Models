import tensorflow as tf
from keras_radam import RAdam
from graphs.disentangled_inferred_prior.AE_graph import create_Bayesian_losses
from evaluation.quantitive_metrics.metrics import create_metrics
from training.autoencoding_basic.autoencoders.autoencoder import autoencoder as basicAE
from training.disentangled_inferred_prior.DIP_shared import infer_prior
import tensorflow_probability as tfp
from utils.swe.codes import epsilon

class Bayesian_Covariance_AE(basicAE):
    # override function
    def compile(
            self,
            optimizer=RAdam(),
            loss=None,
            **kwargs
    ):

        ae_losses = create_Bayesian_losses()
        loss = loss or {}
        for k in loss:
            ae_losses.pop(k)
        self.ae_losses = {**ae_losses, **loss}

        if 'metrics' in kwargs.keys():
            self.ae_metrics = kwargs.pop('metrics', None)
        else:
            self.ae_metrics = create_metrics([self.batch_size] + self.get_inputs_shape()[-3:])

        tf.keras.Model.compile(self, optimizer=optimizer, loss=self.ae_losses, metrics=self.ae_metrics, **kwargs)
        print(self.summary())

    def __encode__(self, **kwargs):
        inputs = kwargs['inputs']
        for k, v in  inputs.items():
            if inputs[k].shape == self.get_inputs_shape():
                inputs[k] = tf.reshape(inputs[k], (1, ) + self.get_inputs_shape())
            inputs[k] = tf.cast(inputs[k], tf.float32)
        kwargs['model']  = self.get_variable
        kwargs['latents_shape'] = (self.batch_size, self.latents_dim)

        encoded = self.encode_fn(**kwargs)
        covariance_mean  = infer_prior(latent_mean=encoded['z_latents'], \
                                       regularize=False, lambda_d=self.lambda_d, lambda_od=self.lambda_od)

        covariance_sigma = tf.exp(encoded['z_latents']+epsilon)
        latents_sigma = covariance_sigma
        prior_distribution = tfp.distributions.Normal(loc=covariance_mean, scale=covariance_sigma)
        posterior_distribution = tfp.distributions.Normal(loc=encoded['z_latents'], scale=latents_sigma)
        bayesian_divergent = tfp.distributions.kl_divergence(posterior_distribution, prior_distribution)
        bayesian_divergent = tf.identity(bayesian_divergent, name='bayesian_divergent' )
        return {**encoded,
                'bayesian_divergent': bayesian_divergent}

    def __init_autoencoder__(self, **kwargs):
        #  disentangled_inferred_prior configuration
        self.lambda_d = 100
        self.lambda_od = 50

        # connect the graph x' = decode(encode(x))
        inputs_dict= {k: v.inputs[0] for k, v in self.get_variables().items() if k == 'inference'}
        encoded = self.__encode__(inputs=inputs_dict)
        x_logits = self.decode(latents={'z_latents': encoded['z_latents']})
        bayesian_divergent = encoded['bayesian_divergent']

        outputs_dict = {
            'x_logits': x_logits,
            'bayesian_divergent': bayesian_divergent
        }

        tf.keras.Model.__init__(
            self,
            name=self.name,
            inputs=inputs_dict,
            outputs=outputs_dict,
            **kwargs
        )

    def __rename_outputs__(self):
        # rename the outputs
        ## rename the outputs
        for i, output_name in enumerate(self.output_names):
            if 'x_logits' in output_name:
                self.output_names[i] = 'x_logits'
            elif 'bayesian_divergent' in output_name:
                self.output_names[i] = 'bayesian_divergent'
            else:
                pass

    def batch_cast(self, batch):
        if self.input_kw:
            x = tf.cast(batch[self.input_kw], dtype=tf.float32)/self.input_scale
        else:
            x = tf.cast(batch, dtype=tf.float32)/self.input_scale

        return {
                   'inference_inputs': x,
               }, \
               {
                   'x_logits': x,
                   'bayesian_divergent': 0.0
               }

