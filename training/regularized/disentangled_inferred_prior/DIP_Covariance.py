import tensorflow as tf
from training.autoencoding_basic.autoencoders.autoencoder import autoencoder as basicAE

class DIP_Cov(basicAE):
    def __init_autoencoder__(self, **kwargs):
        #  DIP configuration
        self.lambda_d = 5
        self.d_factor = 5
        self.d = self.d_factor * self.lambda_d

        # connect the graph x' = decode(encode(x))
        inputs_dict= {k: v.inputs[0] for k, v in self.get_variables().items() if k == 'inference'}
        latents = self.__encode__(inputs=inputs_dict)
        x_logits = self.decode(latents)

        regularizer = self.create_regularizer(latents['z_latents'])
        outputs_dict = {
            'x_logits': x_logits,
            #'regularizer': regularizer
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
            elif 'regularizer' in output_name:
                self.output_names[i] = 'regularizer'
            else:
                print(self.output_names[i])


    '''
    ------------------------------------------------------------------------------
                                         DIP_Covarance OPERATIONS
    ------------------------------------------------------------------------------
    '''
    def create_regularizer(self, latent_mean, latent_logvar=None):
        cov_latent_mean = self.compute_covariance_latent_mean(latent_mean)

        # Eq 6 page 4
        # mu = z_mean is [batch_size, num_latent]
        # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
        cov_dip_regularizer = tf.add(self.regularize_diag_off_diag_dip(cov_latent_mean, self.lambda_d, self.d), 0.0,
                                     name='regularizer')

        return cov_dip_regularizer

    def compute_covariance_latent_mean(self, latent_mean):
        """
        :param latent_mean:
        :return:
        Computes the covariance of latent_mean.
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
        Compute on and off diagonal regularizers for DIP_Covarance-VAE models.
        Penalize deviations of covariance_matrix from the identity matrix. Uses
        different weights for the deviations of the diagonal and off diagonal entries.
        Args:
            covariance_matrix: Tensor of size [num_latent, num_latent] to covar_reg.
            lambda_od: Weight of penalty for off diagonal elements.
            lambda_d: Weight of penalty for diagonal elements.
        Returns:
            dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
        """
        covariance_matrix_diagonal = tf.linalg.diag_part(covariance_matrix)
        covariance_matrix_off_diagonal = covariance_matrix - tf.linalg.diag(covariance_matrix_diagonal)
        dip_regularizer = tf.add(
              lambda_od * tf.reduce_sum(covariance_matrix_off_diagonal**2),
              lambda_d * tf.reduce_sum((covariance_matrix_diagonal - 1)**2))

        return dip_regularizer
