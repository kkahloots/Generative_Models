import tensorflow as tf
from keras_radam import RAdam
from graphs.regularized.DIP_VAE_graph import create_losses
from evaluation.quantitive_metrics.metrics import create_metrics
from training.autoencoding_basic.autoencoders.VAE import VAE as VAE
from statistical.pdfs import log_normal_pdf

class DIP_Cov_VAE(VAE):

    # override function
    def compile(
            self,
            optimizer=RAdam(),
            loss=None,
            **kwargs
    ):

        ae_losses = create_losses()
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
        #  DIP configuration
        self.lambda_d = 5
        self.d_factor = 5
        self.d = self.d_factor * self.lambda_d

        # mean, logvariance = self.__encode__(inputs)
        # z = reparametrize(mean, logvariance)
        # connect the graph x' = decode(z)
        inputs_dict= {
            'x_mean': self.get_variables()['inference_mean'].inputs[0],
            'x_logvariance': self.get_variables()['inference_logvariance'].inputs[0]
        }
        encoded = self.__encode__(inputs=inputs_dict)
        x_logits = self.decode(encoded['z_latents'])
        covariance_regularizer = encoded['covariance_regularized']

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
            'covariance_regularized': covariance_regularizer
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
            elif 'covariance_regularized' in output_name:
                self.output_names[i] = 'covariance_regularized'
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
        covariance_regularizer = self.regularize(encoded['z_latents'])
        return {**encoded, 'covariance_regularized': covariance_regularizer}


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
                   'x_logpdf':0.0,
                   'covariance_regularized': 0.0
               }


    '''
    ------------------------------------------------------------------------------
                                         DIP_Covarance OPERATIONS
    ------------------------------------------------------------------------------
    '''
    def regularize(self, latent_mean, latent_logvar=None):
        cov_latent_mean = self.compute_covariance_latent_mean(latent_mean)

        # Eq 6 page 4
        # mu = z_mean is [batch_size, num_latent]
        # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
        cov_dip_regularizer = self.regularize_diag_off_diag_dip(cov_latent_mean, self.lambda_d, self.d)
        cov_dip_regularizer = tf.add(cov_dip_regularizer, 0.0, name='covariance_regularized')
        return cov_dip_regularizer

    def compute_covariance_latent_mean(self, latent_mean):
        """
        :param latent_mean:
        :return:
        Computes the covariance_regularizer of latent_mean.
        Uses cov(latent_mean) = E[latent_mean*latent_mean^T] - E[latent_mean]E[latent_mean]^T.
        Args:
          latent_mean: Encoder mean, tensor of size [batch_size, num_latent].
        Returns:
          cov_latent_mean: Covariance of encoder mean, tensor of size [latent_dim, latent_dim].
        """
        exp_latent_mean_latent_mean_t = tf.reduce_mean(
            tf.expand_dims(latent_mean, 2) * tf.expand_dims(latent_mean, 1), axis=0)
        expectation_latent_mean = tf.reduce_mean(latent_mean, axis=0)

        cov_latent_mean = tf.subtract(exp_latent_mean_latent_mean_t,
          tf.expand_dims(expectation_latent_mean, 1) * tf.expand_dims(expectation_latent_mean, 0))
        return cov_latent_mean

    def regularize_diag_off_diag_dip(self, covariance_matrix, lambda_od, lambda_d):
        """
        Compute on and off diagonal covariance_regularizer for DIP_Covarance-VAE models.
        Penalize deviations of covariance_matrix from the identity matrix. Uses
        different weights for the deviations of the diagonal and off diagonal entries.
        Args:
            covariance_matrix: Tensor of size [num_latent, num_latent] to covar_reg.
            lambda_od: Weight of penalty for off diagonal elements.
            lambda_d: Weight of penalty for diagonal elements.
        Returns:
            dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
        """
        #matrix_diag_part
        covariance_matrix_diagonal = tf.linalg.diag_part(covariance_matrix)
        covariance_matrix_off_diagonal = covariance_matrix - tf.linalg.diag(covariance_matrix_diagonal)
        dip_regularizer = tf.add(
              lambda_od * covariance_matrix_off_diagonal**2,
              lambda_d * (covariance_matrix_diagonal - 1)**2)
        return dip_regularizer
