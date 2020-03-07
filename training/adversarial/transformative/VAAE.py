import tensorflow as tf

from graphs.adversarial_graph.AAE_graph import latent_discriminate_encode_fn
from stats.adver_losses import create_adversarial_real_losses, create_adversarial_fake_losses, create_adversarial_losses
from training.traditional.transformative.VAE import VAE as autoencoder
from utils.swe.codes import copy_fn


class VAAE(autoencoder):
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

    def get_discriminators(self):
        return {
            'latent_real_discriminator': self.latent_real_discriminator,
            'latent_fake_discriminator': self.latent_fake_discriminator
        }


    def latent_real_discriminator_cast_batch(self,  xt0, xt1):
        xt0 = tf.cast(xt0, dtype=tf.float32) / self.input_scale
        xt1 = tf.cast(xt1, dtype=tf.float32) / self.input_scale

        en = autoencoder.encode(self, inputs={'inputs': xt0})
        return {'generative_inputs': en['x_latent'],
                'latent_real_discriminator_inputs': en['x_latent']
                } ,\
               {
                   'latent_real_discriminator_outputs': tf.ones(shape=[self.batch_size, 1], name='real_true')
               }

    def latent_fake_discriminator_cast_batch(self,  xt0, xt1):
        xt0 = tf.cast(xt0, dtype=tf.float32) / self.input_scale
        xt1 = tf.cast(xt1, dtype=tf.float32) / self.input_scale

        en = autoencoder.encode(self, inputs={'inputs': xt0})
        return {'generative_inputs': en['x_latent'],
                'latent_fake_discriminator_inputs': en['x_latent']
                } ,\
               {
                   'latent_fake_discriminator_outputs': tf.ones(shape=[self.batch_size, 1], name='fake_true')
               }

    def together_cast_batch(self,  xt0, xt1):
        xt0 = tf.cast(xt0, dtype=tf.float32) / self.input_scale
        xt1 = tf.cast(xt1, dtype=tf.float32) / self.input_scale

        en = autoencoder.encode(self, inputs={'inputs': xt0})
        return {   'inference_inputs': xt0,
                   'generative_inputs': en['x_latent']} ,\
               {
                   'x_logits': xt1,
                   'latent_real_discriminator_outputs': tf.ones(shape=[self.batch_size, 1], name='real_true'),
                   'latent_fake_discriminator_outputs': tf.zeros(shape=[self.batch_size, 1], name='fake_true')
               }

    def fit(
            self,
            x,
            y=None,
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
        autoencoder.fit(
            self,
            x=x,
            y=y,
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

            temp_layers = tf.keras.models.clone_model(self.get_variables()['generative']).layers
            temp_layers.append(tf.keras.layers.Flatten())
            temp_layers.append(tf.keras.layers.Dense(units=1, activation='sigmoid', name='latent_real_discriminator_outputs'))
            temp_layers = tf.keras.Sequential(temp_layers)
            self.latent_real_discriminator = tf.keras.Model(
                name='latent_real_discriminator',
                inputs=temp_layers.inputs,
                outputs=temp_layers.outputs
            )

            temp_layers = tf.keras.models.clone_model(self.get_variables()['generative']).layers
            temp_layers.append(tf.keras.layers.Flatten())
            temp_layers.append(tf.keras.layers.Dense(units=1, activation='sigmoid', name='latent_fake_discriminator_outputs'))
            temp_layers = tf.keras.Sequential(temp_layers)
            self.latent_fake_discriminator = tf.keras.Model(
                name='latent_fake_discriminator',
                inputs=temp_layers.inputs,
                outputs=temp_layers.outputs
            )

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
        self.latent_real_discriminator.fit(
            x=x.map(self.latent_real_discriminator_cast_batch),
            y=y,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=None,
            validation_data=validation_data.map(self.latent_real_discriminator_cast_batch),
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            class_weight=class_weight,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
            initial_epoch=initial_epoch
        )

        self.latent_fake_discriminator.fit(
            x=x.map(self.latent_fake_discriminator_cast_batch),
            y=y,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=None,
            validation_data=validation_data.map(self.latent_fake_discriminator_cast_batch),
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

        # 7- training together
        self.latent_AA.fit(
            x=x.map(self.together_cast_batch),
            y=y,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=None,
            validation_data=validation_data.map(self.together_cast_batch),
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            class_weight=class_weight,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
            initial_epoch=initial_epoch
        )


    def connect_together(self):
        self.get_variables = self.adver_get_variables
        self.encode_fn = latent_discriminate_encode_fn
        _inputs = {
            'inputs': self.get_variables()['inference'].inputs[0]
        }
        encoded = self.encode(inputs=_inputs)
        x_logits = self.decode(encoded['x_latent'])

        _outputs = {
            'x_logits': x_logits,
            'real_pred': encoded['real_pred'],
            'fake_pred': encoded['fake_pred']
        }
        self.latent_AA = tf.keras.Model(
            name='latent_AA',
            inputs= _inputs,
            outputs=_outputs
        )

        for i, _output in enumerate(self.latent_AA.output_names):
            if 'tf_op_layer_x_logits' in _output :
                self.latent_AA.output_names[i] = 'x_logits'
            elif 'latent_fake_discriminator' in _output :
                self.latent_AA.output_names[i] = 'latent_fake_discriminator_outputs'
            elif 'latent_real_discriminator' in _output :
                self.latent_AA.output_names[i] = 'latent_real_discriminator_outputs'
            else:
                pass

        self.latent_AA.compile(
            optimizer=self.optimizer,
            loss=create_adversarial_losses(),
            metrics=self.temp_metrics
        )

        print(self.latent_AA.summary())

    def latent_discriminator_compile(self, **kwargs):
        self.latent_real_discriminator.compile(
            optimizer=self.optimizer,
            loss=create_adversarial_real_losses(),
            metrics=None
        )

        print(self.latent_real_discriminator.summary())

        self.latent_fake_discriminator.compile(
            optimizer=self.optimizer,
            loss=create_adversarial_fake_losses(),
            metrics=None
        )

        print(self.latent_fake_discriminator.summary())

    # combined models special
    def adver_get_variables(self):
        return {**self.ae_get_variables(), **self.get_discriminators()}


    def adver_loss_functions(self):
        return {**self.adaptee_ae.loss_functions(), **self.get_discriminator_losses()}

