import tensorflow as tf
from keras_radam import RAdam
from graphs.disentangled_inferred_prior.VAE_graph import create_regularized_Bayesian_losses
from evaluation.quantitive_metrics.metrics import create_metrics
from training.autoencoding_basic.autoencoders.VAE import VAE as VAE
from training.disentangled_inferred_prior.DIP_shared import infer_prior
from statistical.pdfs import log_normal_pdf
import tensorflow_probability as tfp
from utils.swe.codes import epsilon

class Bayesian_Covariance_VAE(VAE):
    # override function
    def compile(
            self,
            optimizer=RAdam(),
            loss=None,
            **kwargs
    ):

        ae_losses = create_regularized_Bayesian_losses()
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

    def __init_autoencoder__(self, **kwargs):
        #  disentangled_inferred_prior configuration
        self.lambda_d = 100
        self.lambda_od = 50

        # mean, logvariance = self.__encode__(inputs)
        # z = reparametrize(mean, logvariance)
        # connect the graph x' = decode(z)
        inputs_dict= {
            'x_mean': self.get_variables()['inference_mean'].inputs[0],
            'x_logvariance': self.get_variables()['inference_logvariance'].inputs[0]
        }
        encoded = self.__encode__(inputs=inputs_dict)
        x_logits = self.decode(latents={'z_latents': encoded['z_latents']})
        covariance_regularizer =  encoded['covariance_regularized']
        bayesian_divergent = encoded['bayesian_divergent']

        logpdf = log_normal_pdf(
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
            'logpdf': logpdf,
            'covariance_regularized': covariance_regularizer,
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
            elif 'covariance_regularized' in output_name:
                self.output_names[i] = 'covariance_regularized'
            elif 'bayesian_divergent' in output_name:
                self.output_names[i] = 'bayesian_divergent'
            else:
                pass

    def __encode__(self, **kwargs):
        inputs = kwargs['inputs']
        for k, v in  inputs.items():
            if inputs[k].shape == self.get_inputs_shape():
                inputs[k] = tf.reshape(inputs[k], (1, ) + self.get_inputs_shape())
            inputs[k] = tf.cast(inputs[k], tf.float32)
        kwargs['model']  = self.get_variable
        kwargs['latents_shape'] = (self.batch_size, self.latents_dim)

        encoded = self.encode_fn(**kwargs)
        covariance_mean, covariance_regularizer = infer_prior(latent_mean=encoded['inference_mean'], \
                                                              regularize=True, lambda_d=self.lambda_d, lambda_od=self.lambda_od)

        covariance_sigma = tf.exp(encoded['inference_logvariance']+ epsilon)
        latents_sigma = covariance_sigma
        prior_distribution = tfp.distributions.Normal(loc=covariance_mean, scale=covariance_sigma)
        posterior_distribution = tfp.distributions.Normal(loc=encoded['inference_mean'], scale=latents_sigma)
        bayesian_divergent = tfp.distributions.kl_divergence(posterior_distribution, prior_distribution)
        bayesian_divergent = tf.identity(bayesian_divergent, name='bayesian_divergent' )
        return {**encoded,
                'covariance_regularized': covariance_regularizer,
                'bayesian_divergent': bayesian_divergent}

    def batch_cast(self, batch):
        if self.input_kw:
            x = tf.cast(batch[self.input_kw], dtype=tf.float32)/self.input_scale
        else:
            x = tf.cast(batch, dtype=tf.float32)/self.input_scale

        return {
                   'inference_logvariance_inputs': x,
                   'inference_mean_inputs': x
               }, \
               {
                   'x_logits': x,
                   'z_latents': 0.0,
                   'x_logpdf':0.0,
                   'covariance_regularized': 0.0,
                   'bayesian_divergent': 0.0
               }

