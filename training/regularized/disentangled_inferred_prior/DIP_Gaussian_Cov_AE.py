import tensorflow as tf
from training.regularized.disentangled_inferred_prior.DIP_Covariance_AE import DIP_Cov_AE
class DIP_Gaussian_Cov_AE(DIP_Cov_AE):
    def __encode__(self, **kwargs):
        inputs = kwargs['inputs']
        for k, v in  inputs.items():
            if inputs[k].shape == self.get_inputs_shape():
                inputs[k] = tf.reshape(inputs[k], (1, ) + self.get_inputs_shape())
            inputs[k] = tf.cast(inputs[k], tf.float32)
        kwargs['model']  = self.get_variable
        kwargs['latents_shape'] = (self.batch_size, self.latents_dim)

        encoded = self.encode_fn(**kwargs)
        covariance_regularizer = self.regularize(encoded['z_latents'], tf.sigmoid(encoded['z_latents']))
        return {**encoded, 'covariance_regularized': covariance_regularizer}

    '''
    ------------------------------------------------------------------------------
                                         DIP_Covarance OPERATIONS
    ------------------------------------------------------------------------------
    '''
    def regularize(self, latent_mean, latent_logvar):
        cov_latent_mean = self.compute_covariance_latent_mean(latent_mean)
        cov_enc = tf.linalg.diag(tf.exp(latent_logvar))
        expectation_cov_enc = tf.reduce_mean(cov_enc, axis=0)
        cov_latent = expectation_cov_enc + cov_latent_mean

        # Eq 6 page 4
        # mu = z_mean is [batch_size, num_latent]
        # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
        cov_dip_regularizer = self.regularize_diag_off_diag_dip(cov_latent, self.lambda_d, self.d)
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
