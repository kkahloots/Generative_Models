from collections.abc import Iterable
from graphs.adversarial_graph.AAE_graph import latent_discriminate_encode_fn
import tensorflow as tf
import numpy as np

from stats.adver_losses import create_adversarial_losses
from training.traditional.autoencoders.autoencoder import autoencoder
from utils.swe.codes import copy_fn

class AAE(autoencoder):
    def __init__(
            self,
            strategy=None,
            **kwargs
    ):
        self.strategy = strategy
        autoencoder.__init__(
            self,
            **kwargs
        )

    # def get_varibale(self, var_name, param):
    #     if var_name=='latent_discriminator':
    #         return self.get_variables()[var_name].predict(*param)
    #     return self.get_variables()[var_name](*param)

    def get_discriminators(self):
        return dict(zip(['latent_discriminator'], [self.latent_discriminator.call]))

    def latent_discriminator_cast_batch(self, batch):
        x = tf.cast(batch[self.input_kw], dtype=tf.float32) / self.input_scale
        en = autoencoder.encode(self, inputs={'inputs': x})
        return {'generative_inputs': en['x_latent']} ,\
               {
                   'latent_real_outputs': tf.ones(shape=[self.batch_size, 1], name='real_true'),
                   'latent_fake_outputs': tf.zeros(shape=[self.batch_size, 1], name='fake_true')
               }

    def together_cast_batch(self, batch):
        x = tf.cast(batch[self.input_kw], dtype=tf.float32) / self.input_scale
        en = autoencoder.encode(self, inputs={'inputs': x})
        return {'generative_inputs': en['x_latent']} ,\
               {
                   'latent_real_outputs': tf.ones(shape=[self.batch_size, 1], name='real_true'),
                   'latent_fake_outputs': tf.zeros(shape=[self.batch_size, 1], name='fake_true')
               }


    def fit_generator(
            self,
            generator,
            input_kw='image',
            input_scale=1.0,
            steps_per_epoch=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_data=None,
            validation_steps=None,
            validation_freq=1,
            class_weight=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            shuffle=True,
            initial_epoch=0
    ):
        # 1- train the traditional basicAE
        autoencoder.fit_generator(
            self,
            generator=generator,
            input_kw=input_kw,
            input_scale=input_scale,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            class_weight=class_weight,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
            initial_epoch=initial_epoch
        )

        def make_latent_discriminator():
            for k, var in self.get_variables().items():
                for layer in var.layers:
                    if not isinstance(layer, tf.keras.layers.Activation):
                        if hasattr(layer, 'activation'):
                            layer.activation = tf.keras.activations.elu

            discriminator = tf.keras.models.clone_model(self.get_variables()['generative'])
            discriminator.add(tf.keras.layers.Flatten())
            self.latent_discriminator = tf.keras.Model(
                inputs=discriminator.inputs,
                outputs={
                    'latent_real_outputs': tf.keras.layers.Dense(units=1, activation='sigmoid', name='latent_real_outputs')(discriminator.output),
                    'latent_fake_outputs': tf.keras.layers.Dense(units=1, activation='sigmoid', name='latent_fake_outputs')(discriminator.output)
                }, name='latent_discriminator'
            )
            self.latent_discriminator.input._name = 'latent_discriminator_inputs'

        # 2- create a latent discriminator
        if self.strategy:
            with self.strategy:
                make_latent_discriminator()
        else:
            make_latent_discriminator()

        # 3- clone autoencoder variables
        self.ae_get_variables = copy_fn(self.get_variables)

        # 4- switch to discriminate
        if self.strategy:
            if self.strategy:
                self.latent_discriminator_compile()
        else:
            self.latent_discriminator_compile()

        # 5- train the latent discriminator
        self.latent_discriminator.fit_generator(
            generator=generator.map(self.latent_discriminator_cast_batch),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=None,
            validation_data=validation_data,
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            class_weight=class_weight,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
            initial_epoch=initial_epoch
        )

        # 6- connect all for adversarial training
        if self.strategy:
            if self.strategy:
                self.connect_together()
        else:
            self.connect_together()

    def connect_together(self):
        self.get_variables = self.adver_get_variables
        self.encode_fn = latent_discriminate_encode_fn
        _inputs = {
            'inputs': self.get_variables()['inference'].inputs[0]
        }
        encoded = self.encode(inputs=_inputs)
        x_logits = self.decode(encoded['x_latent'])
        #en_outputs = 'real_pred': real_pred {k: v.outputs[0] for k, v in encoded.items()}
        _outputs = {
            'x_logits': x_logits,
            'real_pred': encoded['real_pred']
        }
        self.latent_AA = tf.keras.Model(
            name='latent_AA',
            inputs=_inputs,
            outputs=_outputs
        )
        print(self.latent_AA.summary())

        tf.keras.Model.__init__(
            self,
            name=self.name,
            inputs=_inputs,
            outputs=_outputs
        )


    def latent_discriminator_compile(self, **kwargs):
        #self.get_variables = self.get_discriminators
        #self.encode_fn = latent_discriminate_encode_fn
        self.latent_discriminator.compile(
            optimizer=self.optimizer,
            loss=create_adversarial_losses(),
            metrics=None
        )
        print(self.latent_discriminator.summary())

    # combined models special
    def adver_get_variables(self):
        return {**self.ae_get_variables(), **self.get_discriminators()}

        #self.adaptee_ae.loss_functions = self.get_discriminator_losses

    # def copy_adaptee(self):
    #     self.temp_get_variables = copy_fn(self.adaptee_ae.get_variables)
    #     self.temp_feedforwad = copy_fn(self.adaptee_ae.feedforwad)
    #     self.temp_loss_functions = copy_fn(self.adaptee_ae.loss_functions)
    #
    # def switch_2tranditional(self):
    #     self.adaptee_ae.get_variables = self.temp_get_variables
    #     self.adaptee_ae.feedforwad = self.temp_feedforwad
    #     self.adaptee_ae.loss_functions = self.temp_loss_functions
    #
    # discriminator special

    def discriminator_feedforwad(self, inputs):
        return {dkey: self.adver_feedforwad(inputs)[dkey] for dkey in ['real_pred', 'fake_pred']}

    def adver_feedforwad(self, inputs):
        ae_output = self.adaptee_ae.feedforwad(inputs)

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
        self.adaptee_ae.feedforwad = self.adver_feedforwad
        self.adaptee_ae.loss_functions = self.adver_loss_functions


    def fit1(self, train_dataset, test_dataset,
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
        self.latent_discriminator_compile()

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

