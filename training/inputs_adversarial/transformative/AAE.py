from graphs.adversarial_graph.AAE_graph import logits_discriminate_encode_fn
import tensorflow as tf

from stats.adver_losses import create_latent_adversarial_real_losses, create_latent_adversarial_fake_losses, create_latent_adversarial_losses
from training.traditional.transformative.AE import autoencoder
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

    def get_discriminators(self):
        return {
            'logits_real_discriminator': self.logits_real_discriminator,
            'logits_fake_discriminator': self.logits_fake_discriminator
        }

    def logits_real_discriminator_cast_batch(self, xt0, xt1):
        xt0 = tf.cast(xt0, dtype=tf.float32) / self.input_scale
        xt1 = tf.cast(xt1, dtype=tf.float32) / self.input_scale

        en = autoencoder.encode(self, inputs={'inputs': xt0})
        return {'generative_inputs': en['x_latent'],
                'logits_real_discriminator_inputs': en['x_latent']
                } ,\
               {
                   'logits_real_discriminator_outputs': tf.ones(shape=[self.batch_size]+[self.inputs_shape], name='real_true')
               }

    def logits_fake_discriminator_cast_batch(self, xt0, xt1):
        xt0 = tf.cast(xt0, dtype=tf.float32) / self.input_scale
        xt1 = tf.cast(xt1, dtype=tf.float32) / self.input_scale

        en = autoencoder.encode(self, inputs={'inputs': xt0})
        return {'generative_inputs': en['x_latent'],
                'logits_fake_discriminator_inputs': en['x_latent']
                } ,\
               {
                   'logits_fake_discriminator_outputs': tf.zeros(shape=[self.batch_size]+[self.inputs_shape], name='fake_true')
               }

    def together_cast_batch(self,  xt0, xt1):
        xt0 = tf.cast(xt0, dtype=tf.float32) / self.input_scale
        xt1 = tf.cast(xt1, dtype=tf.float32) / self.input_scale

        en = autoencoder.encode(self, inputs={'inputs': xt0})
        return {   'inference_inputs': xt0,
                   'generative_inputs': en['x_latent']} ,\
               {
                   'x_logits': xt1,
                   'logits_real_discriminator_outputs': tf.ones(shape=[self.batch_size]+[self.inputs_shape], name='real_true'),
                   'logits_fake_discriminator_outputs': tf.zeros(shape=[self.batch_size]+[self.inputs_shape], name='fake_true')
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
        print()
        print('training traditional basicAE')
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

        def make_logits_discriminator():
            for k, var in self.get_variables().items():
                for layer in var.layers:
                    if not isinstance(layer, tf.keras.layers.Activation):
                        if hasattr(layer, 'activation'):
                            layer.activation = tf.keras.activations.elu

            temp_layers = tf.keras.models.clone_model(self.get_variables()['generative']).layers
            temp_layers.append(tf.keras.layers.Flatten())
            temp_layers.append(tf.keras.layers.Dense(units=1, activation='sigmoid', name='logits_real_discriminator_outputs'))
            temp_layers = tf.keras.Sequential(temp_layers)
            self.logits_real_discriminator = tf.keras.Model(
                name='logits_real_discriminator',
                inputs=temp_layers.inputs,
                outputs=temp_layers.outputs
            )

            temp_layers = tf.keras.models.clone_model(self.get_variables()['generative']).layers
            temp_layers.append(tf.keras.layers.Flatten())
            temp_layers.append(tf.keras.layers.Dense(units=1, activation='sigmoid', name='logits_fake_discriminator_outputs'))
            temp_layers = tf.keras.Sequential(temp_layers)
            self.logits_fake_discriminator = tf.keras.Model(
                name='logits_fake_discriminator',
                inputs=temp_layers.inputs,
                outputs=temp_layers.outputs
            )

        # 2- create a logits discriminator
        if self.strategy:
            with self.strategy:
                make_logits_discriminator()
        else:
            make_logits_discriminator()

        # 3- clone autoencoder variables
        self.ae_get_variables = copy_fn(self.get_variables)

        # 4- switch to discriminate
        if self.strategy:
            if self.strategy:
                self.logits_discriminator_compile()
        else:
            self.logits_discriminator_compile()

        print()
        print('training logits real discriminator')
        # 5- train the logits discriminator
        self.logits_real_discriminator.fit(
            x=x.map(self.logits_real_discriminator_cast_batch),
            y=y,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=1,
            callbacks=None,
            validation_data=validation_data.map(self.logits_real_discriminator_cast_batch),
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            class_weight=class_weight,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
            initial_epoch=initial_epoch
        )

        print()
        print('training logits fake discriminator')
        self.logits_fake_discriminator.fit(
            x=x.map(self.logits_fake_discriminator_cast_batch),
            y=y,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=1,
            callbacks=None,
            validation_data=validation_data.map(self.logits_fake_discriminator_cast_batch),
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
            cb.keys = ['loss'] + [fn+'_loss' for fn in self.logits_AA.output_names] + mertic_names
            cb.append_header = cb.keys

        # 7- training together
        self.logits_AA.fit(
            x=x.map(self.together_cast_batch),
            y=y,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=0,
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
        self.encode_fn = logits_discriminate_encode_fn
        _inputs = {
            'inputs': self.get_variables()['inference'].inputs[0]
        }
        encoded = self.encode(inputs=_inputs)
        x_logits = self.decode(encoded['x_latent'])

        _outputs = {
            'x_logits': x_logits,
            'logits_real_pred': encoded['logits_real_pred'],
            'logits_fake_pred': encoded['logits_fake_pred']
        }
        self.logits_AA = tf.keras.Model(
            name='logits_AA',
            inputs= _inputs,
            outputs=_outputs
        )

        for i, _output in enumerate(self.logits_AA.output_names):
            if 'tf_op_layer_x_logits' in _output :
                self.logits_AA.output_names[i] = 'x_logits'
            elif 'logits_fake_discriminator' in _output :
                self.logits_AA.output_names[i] = 'logits_fake_discriminator_outputs'
            elif 'logits_real_discriminator' in _output :
                self.logits_AA.output_names[i] = 'logits_real_discriminator_outputs'
            else:
                pass

        self.logits_AA.compile(
            optimizer=self.optimizer,
            loss=create_latent_adversarial_losses(),
            metrics=self.temp_metrics
        )
        self.logits_AA.generate_sample = self.generate_sample
        self.logits_AA.get_varibale = self.get_varibale
        self.logits_AA.inputs_shape = self.inputs_shape
        self.logits_AA.latent_dim = self.latent_dim

        print(self.logits_AA.summary())


    def logits_discriminator_compile(self, **kwargs):
        self.logits_real_discriminator.compile(
            optimizer=self.optimizer,
            loss=create_latent_adversarial_real_losses(),
            metrics=None
        )

        print(self.logits_real_discriminator.summary())

        self.logits_fake_discriminator.compile(
            optimizer=self.optimizer,
            loss=create_latent_adversarial_fake_losses(),
            metrics=None
        )

        print(self.logits_fake_discriminator.summary())

    # combined models special
    def adver_get_variables(self):
        return {**self.ae_get_variables(), **self.get_discriminators()}

    def adver_loss_functions(self):
        return {**self.adaptee_ae.loss_functions(), **self.get_discriminator_losses()}
