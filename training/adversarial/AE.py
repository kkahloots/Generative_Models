from collections.abc import Iterable

import tensorflow as tf

from training.adversarial.losses import compute_discr_bce
from training.traditional.autoencoders.AE import AE as AdapteeAE
from utils.swe.codes import copy_func


class AE():
    def __init__(
            self,
            model_name,
            inputs_shape,
            outputs_shape,
            latent_dim,
            variables_params,
            restore=None,
            AE=AdapteeAE
    ):

        self.adaptee_ae = AE(
                            model_name=model_name,
                            inputs_shape=inputs_shape,
                            outputs_shape=outputs_shape,
                            latent_dim=latent_dim,
                            variables_params=variables_params,
                            restore=restore
                        )

    def copy_adaptee(self):
        self.temp_get_variables = copy_func(self.adaptee_ae.get_variables)
        self.temp_feedforward = copy_func(self.adaptee_ae.feedforward)
        self.temp_loss_functions = copy_func(self.adaptee_ae.loss_functions)

    def switch_2tranditional(self):
        self.adaptee_ae.get_variables = self.temp_get_variables
        self.adaptee_ae.feedforward = self.temp_feedforward
        self.adaptee_ae.loss_functions = self.temp_loss_functions

    # discriminator special
    def switch_2discriminate(self):
        self.adaptee_ae.get_variables = self.get_discriminators
        self.adaptee_ae.feedforward = self.discriminator_feedforward
        self.adaptee_ae.loss_functions = self.get_discriminator_losses

    def get_discriminator_losses(self):
        return {'discr_bce': compute_discr_bce}

    def get_discriminators(self):
        return dict(zip(['discriminator'], [self.adaptee_ae.discriminator]))

    def discriminator_feedforward(self, inputs):
        return {dkey: self.adver_feedforward(inputs)[dkey] for dkey in ['real_pred', 'fake_pred']}

    # combined models special
    def adver_get_variables(self):
        return {**self.adaptee_ae.get_variables(), **self.get_discriminators()}

    def adver_feedforward(self, inputs):
        ae_output = self.adaptee_ae.feedforward(inputs)

        # swapping the true by random
        fake_latent = ae_output['latent']
        real_latent = tf.random.normal(shape=fake_latent.shape)
        real_pred = self.adaptee_ae.discriminator(real_latent)
        fake_pred = self.adaptee_ae.discriminator(fake_latent)
        return {**ae_output, 'real_pred': real_pred, 'fake_pred': fake_pred}

    def adver_loss_functions(self):
        return {**self.adaptee_ae.loss_functions(), **self.get_discriminator_losses()}

    def switch_2adversarial(self):
        self.adaptee_ae.get_variables = self.adver_get_variables
        self.adaptee_ae.feedforward = self.adver_feedforward
        self.adaptee_ae.loss_functions = self.adver_loss_functions


    def fit(self, train_dataset, test_dataset,
            instance_names=['image'],
            epochs=10,
            learning_rate=1e-3,
            random_latent=None,
            recoding_dir='./recoding',
            gray_plot=True,
            generate_epoch=5,
            save_epoch=5,
            metric_epoch=10,
            gt_epoch=10,
            gt_data=None
            ):
        assert isinstance(train_dataset, Iterable), 'dataset must be iterable'
        assert isinstance(test_dataset, Iterable), 'dataset must be iterable'

        # 1- train the traditional basicAE
        self.adaptee_ae.fit(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            instance_names=instance_names,
            epochs=epochs,
            learning_rate=learning_rate,
            random_latent=random_latent,
            recoding_dir=recoding_dir,
            gray_plot=gray_plot,
            generate_epoch=None,
            save_epoch=save_epoch,
            metric_epoch=None,
            gt_epoch=None,
            gt_data=None)

        # 2- copy traditional
        self.copy_adaptee()

        # 3- create a discriminator
        for var in self.adaptee_ae.get_variables():
            for layer in var.layers:
                if not isinstance(layer, tf.keras.layers.Activation):
                    if hasattr(layer, 'activation'):
                        layer.activation = tf.keras.activations.elu

        self.adaptee_ae.discriminator = tf.keras.models.clone_model(self.adaptee_ae.get_variables()['generative'])
        self.adaptee_ae.discriminator._name = 'discriminator'
        self.adaptee_ae.discriminator.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        print(self.adaptee_ae.discriminator.summary())

        # 4- switch to discriminate
        self.switch_2discriminate()

        # 5- train the discriminator
        self.adaptee_ae.fit(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            instance_names=instance_names,
            epochs=epochs,
            learning_rate=learning_rate,
            random_latent=random_latent,
            recoding_dir=recoding_dir,
            gray_plot=gray_plot,
            generate_epoch=None,
            save_epoch=save_epoch,
            metric_epoch=None,
            gt_epoch=None,
            gt_data=None)

        # 6- switch to adversarial
        self.switch_2adversarial()
        self.adaptee_ae.fit(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            instance_names=instance_names,
            epochs=epochs,
            learning_rate=learning_rate,
            random_latent=random_latent,
            recoding_dir=recoding_dir,
            gray_plot=gray_plot,
            generate_epoch=generate_epoch,
            save_epoch=save_epoch,
            metric_epoch=metric_epoch,
            gt_epoch=gt_epoch,
            gt_data=gt_data)
