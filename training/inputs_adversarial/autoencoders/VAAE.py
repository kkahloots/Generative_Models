import tensorflow as tf
from training.callbacks.early_stopping import EarlyStopping

from graphs.adversarial_graph.VAAE_graph import inputs_discriminate_encode_fn
from stats.adver_losses import create_inputs_adversarial_real_losses, create_inputs_adversarial_fake_losses, \
    create_inputs_adversarial_losses
from training.traditional.autoencoders.VAE import VAE as autoencoder
from utils.swe.codes import copy_fn


class VAAE(autoencoder):
    def __init__(
            self,
            inputs_adversarial_losses,
            strategy=None,
            **kwargs
    ):
        self.inputs_adversarial_losses = inputs_adversarial_losses
        self.strategy = strategy
        autoencoder.__init__(
            self,
            **kwargs
        )

    def get_discriminators(self):
        return {
            'inputs_real_discriminator': self.inputs_real_discriminator,
            'inputs_fake_discriminator': self.inputs_fake_discriminator
        }


    def inputs_real_discriminator_cast_batch(self, batch):
        if self.input_kw:
            x = tf.cast(batch[self.input_kw], dtype=tf.float32) / self.input_scale
        else:
            x = tf.cast(batch, dtype=tf.float32) / self.input_scale
        en = autoencoder.encode(self, inputs={'x_mean': x, 'x_logvar': x})
        return {'generative_inputs': en['x_latent'],
                'inputs_real_discriminator_inputs': x,
                'encoder_mean_inputs': x,
                'encoder_logvar_inputs': x,
                } ,\
               {
                   'inputs_real_discriminator_outputs': tf.ones(shape=[self.batch_size, 1], name='inputs_real_true')
               }

    def inputs_fake_discriminator_cast_batch(self, batch):
        if self.input_kw:
            x = tf.cast(batch[self.input_kw], dtype=tf.float32) / self.input_scale
        else:
            x = tf.cast(batch, dtype=tf.float32) / self.input_scale
        en = autoencoder.encode(self, inputs={'x_mean': x, 'x_logvar': x})
        return {'generative_inputs': en['x_latent'],
                'inputs_fake_discriminator_inputs': x,
                'encoder_mean_inputs': x,
                'encoder_logvar_inputs': x,
                } ,\
               {
                   'inputs_fake_discriminator_outputs': tf.zeros(shape=[self.batch_size, 1], name='inputs_fake_true')
               }

    def together_cast_batch(self, batch):
        if self.input_kw:
            x = tf.cast(batch[self.input_kw], dtype=tf.float32) / self.input_scale
        else:
            x = tf.cast(batch, dtype=tf.float32) / self.input_scale
        en = autoencoder.encode(self, inputs={'x_mean': x, 'x_logvar': x})
        return {
                   'generative_inputs': en['x_latent'],
                   'encoder_mean_inputs': x,
                   'encoder_logvar_inputs': x,
                   } ,\
               {
                   'x_logits': x,
                   'inputs_real_discriminator_outputs': tf.ones(shape=[self.batch_size, 1] , name='inputs_real_true'),
                   'inputs_fake_discriminator_outputs': tf.zeros(shape=[self.batch_size, 1], name='inputs_fake_true')
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

        def make_inputs_discriminator():
            for k, var in self.get_variables().items():
                for layer in var.layers:
                    if not isinstance(layer, tf.keras.layers.Activation):
                        if hasattr(layer, 'activation'):
                            layer.activation = tf.keras.activations.elu

            temp_layers = tf.keras.models.clone_model(self.get_variables()['encoder_mean']).layers
            temp_layers.append(tf.keras.layers.Flatten())
            temp_layers.append(tf.keras.layers.Dense(units=1, activation='sigmoid', name='inputs_real_discriminator_outputs'))
            temp_layers = tf.keras.Sequential(temp_layers)
            self.inputs_real_discriminator = tf.keras.Model(
                name='inputs_real_discriminator',
                inputs=temp_layers.inputs,
                outputs=temp_layers.outputs
            )

            temp_layers = tf.keras.models.clone_model(self.get_variables()['encoder_mean']).layers
            temp_layers.append(tf.keras.layers.Flatten())
            temp_layers.append(tf.keras.layers.Dense(units=1, activation='sigmoid', name='inputs_fake_discriminator_outputs'))
            temp_layers = tf.keras.Sequential(temp_layers)
            self.inputs_fake_discriminator = tf.keras.Model(
                name='inputs_fake_discriminator',
                inputs=temp_layers.inputs,
                outputs=temp_layers.outputs
            )

        # 2- create inputs discriminator
        if self.strategy:
            with self.strategy:
                make_inputs_discriminator()
        else:
            make_inputs_discriminator()

        # 3- clone autoencoder variables
        self.ae_get_variables = copy_fn(self.get_variables)

        # 4- switch to discriminate
        if self.strategy:
            if self.strategy:
                self.inputs_discriminator_compile()
        else:
            self.inputs_discriminator_compile()

        # 5- train the inputs discriminator
        self.inputs_real_discriminator.fit(
            x=x.map(self.inputs_real_discriminator_cast_batch),
            y=y,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=[EarlyStopping()],
            validation_data=validation_data.map(self.inputs_real_discriminator_cast_batch),
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            class_weight=class_weight,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
            initial_epoch=initial_epoch
        )

        self.inputs_fake_discriminator.fit(
            x=x.map(self.inputs_fake_discriminator_cast_batch),
            y=y,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=[EarlyStopping()],
            validation_data=validation_data.map(self.inputs_fake_discriminator_cast_batch),
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            class_weight=class_weight,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
            initial_epoch=initial_epoch
        )

        # 6- connect all for inputs_adversarial training
        if self.strategy:
            if self.strategy:
                self.connect_together()
        else:
            self.connect_together()

        print()
        print('training together')
        cbs = [cb for cb in callbacks or [] if isinstance(cb, tf.keras.callbacks.CSVLogger)]
        for cb in cbs:
            cb.filename = cb.filename.split('.csv')[0] + '_together.csv'
            mertic_names = [fn for sublist in [[k + '_' + fn.__name__ for fn in v] for k, v in self.temp_metrics.items()]
                            for fn in sublist]
            cb.keys = ['loss'] + [fn+'_loss' for fn in self.inputs_AA.output_names] + mertic_names
            cb.append_header = cb.keys

        # 7- training together
        self.inputs_AA.fit(
            x=x.map(self.together_cast_batch),
            y=y,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
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
        self.encode_fn = inputs_discriminate_encode_fn
        _inputs = {
            'x_mean': self.get_variables()['encoder_mean'].inputs[0],
            'x_logvar': self.get_variables()['encoder_logvar'].inputs[0]
        }
        encoded = self.encode(inputs=_inputs)
        x_logits = self.decode(encoded['x_latent'])

        _outputs = {
            'x_logits': x_logits,
            'inputs_real_pred': encoded['inputs_real_pred'],
            'inputs_fake_pred': encoded['inputs_fake_pred']
        }
        self.inputs_AA = tf.keras.Model(
            name='inputs_AA',
            inputs= _inputs,
            outputs=_outputs
        )

        for i, _output in enumerate(self.inputs_AA.output_names):
            if 'tf_op_layer_x_logits' in _output :
                self.inputs_AA.output_names[i] = 'x_logits'
            elif 'inputs_fake_discriminator' in _output :
                self.inputs_AA.output_names[i] = 'inputs_fake_discriminator_outputs'
            elif 'inputs_real_discriminator' in _output :
                self.inputs_AA.output_names[i] = 'inputs_real_discriminator_outputs'
            else:
                pass

        self.inputs_AA.compile(
            optimizer=self.optimizer,
            loss=self.inputs_adversarial_losses['inputs_adversarial_losses'](),
            metrics=self.temp_metrics
        )

        self.inputs_AA.generate_sample = self.generate_sample
        self.inputs_AA.get_varibale = self.get_varibale
        self.inputs_AA.inputs_shape = self.inputs_shape
        self.inputs_AA.latent_dim = self.latent_dim
        print(self.inputs_AA.summary())

    def inputs_discriminator_compile(self, **kwargs):
        self.inputs_real_discriminator.compile(
            optimizer=self.optimizer,
            loss=self.inputs_adversarial_losses['inputs_adversarial_real_losses'](),
            metrics=None
        )

        print(self.inputs_real_discriminator.summary())

        self.inputs_fake_discriminator.compile(
            optimizer=self.optimizer,
            loss=self.inputs_adversarial_losses['inputs_adversarial_fake_losses'](),
            metrics=None
        )

        print(self.inputs_fake_discriminator.summary())

    # combined models special
    def adver_get_variables(self):
        return {**self.ae_get_variables(), **self.get_discriminators()}

