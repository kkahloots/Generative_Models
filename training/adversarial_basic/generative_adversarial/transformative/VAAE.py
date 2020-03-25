import tensorflow as tf

from graphs.adversarial.VAAE_graph import generative_discriminate_encode_fn
from graphs.builder import layer_stuffing, clone_model
from statistical.pdfs import log_normal_pdf
from training.autoencoding_basic.transformative.VAE import VAE as autoencoder
from training.callbacks.early_stopping import EarlyStopping
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
        self.ONES = tf.ones(shape=[self.batch_size, 1])
        self.ZEROS = tf.zeros(shape=[self.batch_size, 1])

        self.adversarial_models = {
            'generative_discriminator_real':
                {
                    'variable': None,
                    'adversarial_item': 'inference',
                    'adversarial_value': self.ONES
                },
            'generative_discriminator_fake':
                {
                    'variable': None,
                    'adversarial_item': 'inference',
                    'adversarial_value': self.ZEROS
                },
            'inference_generator_fake':
                {
                    'variable': None,
                    'adversarial_item': 'inference',
                    'adversarial_value': self.ONES
                }
        }

    # combined models special
    def adversarial_get_variables(self):
        return {**self.ae_get_variables(), **self.get_discriminators()}


    def get_discriminators(self):
        return {k: model['variable'] for k, model in self.adversarial_models.items()}

    def create_batch_cast(self, models):
        def batch_cast_fn(batch):
            if self.input_kw:
                x = tf.cast(batch[self.input_kw], dtype=tf.float32) / self.input_scale
            else:
                x = tf.cast(batch, dtype=tf.float32) / self.input_scale
            outputs_dict =  {k: model['adversarial_value'] for k, model in models.items()}
            outputs_dict = {'x_logits': x, **outputs_dict}

            return {'inference_mean_inputs': x, 'inference_logvariance_inputs': x },outputs_dict

        return batch_cast_fn


    def compile(
            self,
            adversarial_losses,
            adversarial_weights,
            **kwargs
    ):
        self.adversarial_losses=adversarial_losses
        self.adversarial_weights=adversarial_weights
        autoencoder.compile(
            self,
            **kwargs
        )

    def fit(
            self,
            x,
            validation_data=None,
            **kwargs
    ):
        print()
        print(f'training {autoencoder}')
        # 1- train the basic basicAE
        autoencoder.fit(
            self,
            x=x,
            validation_data=validation_data
            **kwargs
        )


        def create_discriminator():
            for model in self.get_variables().values():
                layer_stuffing(model)

            for k, model in self.adversarial_models.items():
                model['variable'] = clone_model(old_model=self.get_variables()[model['adversarial_item']], restore=self.filepath)

        # 2- create a latent discriminator
        if self.strategy:
            with self.strategy:
                create_discriminator()
        else:
            create_discriminator()

        # 3- clone autoencoder variables
        self.ae_get_variables = copy_fn(self.get_variables)

        # 4- switch to discriminate
        if self.strategy:
            if self.strategy:
                self.discriminators_compile()
        else:
            self.discriminators_compile()

        verbose = kwargs['verbose']
        callbacks = kwargs['callbacks']

        for k, model in self.adversarial_models.items():
            print()
            print(f'training {k}')
            # 5- train the latent discriminator
            model['variables'].fit(
                x=x.map(self.create_batch_cast({k: model})),
                validation_data=None if validation_data is None else validation_data.map(self.create_batch_cast({k: model})),
                callbacks=[EarlyStopping()],
                verbose=1,
                **kwargs
            )

        kwargs['verbose'] = verbose
        kwargs['callbacks'] = callbacks

        # 6- connect all for inference_adversarial training
        if self.strategy:
            if self.strategy:
                self.connect_models()
        else:
            self.connect_models()

        print()
        print('training adversarial models')
        cbs = [cb for cb in callbacks or [] if isinstance(cb, tf.keras.callbacks.CSVLogger)]
        for cb in cbs:
            cb.filename = cb.filename.split('.csv')[0] + '_together.csv'
            mertic_names = [fn for sublist in [[k + '_' + fn.__name__ for fn in v] for k, v in self.ae_metrics.items()]
                            for fn in sublist]
            cb.keys = ['loss'] + [fn+'_loss' for fn in self._AA.output_names] + mertic_names
            cb.append_header = cb.keys

        # 7- training together
        self._AA.fit(
            x=x.map(self.create_batch_cast({k: model})),
            validation_data=None if validation_data is None else validation_data.map(
                self.create_batch_cast({k: model})),
            **kwargs
        )

    def connect_models(self):
        self.get_variables = self.adversarial_get_variables
        self.encode_fn = generative_discriminate_encode_fn
        inputs_dict= {
            'x_mean': self.get_variables()['inference_mean'].inputs[0],
            'x_logvariance': self.get_variables()['inference_logvariance'].inputs[0]
        }
        encoded = self.encode(inputs=inputs_dict)
        x_logits = self.decode(encoded['z_latent'])

        logpdf = log_normal_pdf(
            sample=encoded['z_latent'],
            mean=encoded['inference_mean'],
            logvariance=encoded['inference_logvariance']
        )

        outputs_dict = {k+'_predictions': encoded[k+'_predictions'] for k in self.adversarial_models.keys()}
        outputs_dict = {
            'x_logits': x_logits,
            'z_latent': encoded['z_latent'],
            'inference_mean': encoded['inference_mean'],
            'inference_logvariance': encoded['inference_logvariance'],
            'logpdf': logpdf,
            **outputs_dict
        }

        self._AA = tf.keras.Model(
            name='adverasarial_model',
            inputs= inputs_dict,
            outputs=outputs_dict,
        )

        for i, output_dict in enumerate(self.output_names):
            if 'logpdf' in output_dict:
                self.output_names[i] = 'x_logpdf'
            elif 'z_latent' in output_dict:
                self.output_names[i] = 'z_latent'
            elif 'x_logits' in output_dict:
                self.output_names[i] = 'x_logits'
            elif 'logvariance' in output_dict:
                self.output_names[i] = 'inference_logvariance'
            elif 'inference_mean' in output_dict:
                self.output_names[i] = 'inference_mean'
            else:
                pass
            for k in self.adversarial_models.keys():
                if k in outputs_dict:
                    self._AA.output_names[i] = k+'_outputs'

        generator_weight = self.adversarial_weights['generator_weight']
        discriminator_weight = self.adversarial_weights['discriminator_weight']
        generator_losses = [k for k in self.adversarial_losses.keys() if 'generator' in k]
        dlen = len(self.adversarial_losses)-len(generator_losses)
        aeloss_weights = {k: (1-discriminator_weight)*(1-generator_weight)/len(self.ae_losses) for k in self.ae_losses.keys()}
        gloss_weights = {k: (1-discriminator_weight)*(generator_weight)/len(generator_losses) for k in generator_losses}
        discriminator_weights = {k:  discriminator_weight/dlen for k in self.adversarial_losses.keys() if k not in generator_losses}
        self._AA.compile(
            optimizer=self.optimizer,
            loss={**self.ae_losses, **self.adversarial_losses},
            metrics=self.ae_metrics,
            loss_weights={**aeloss_weights, **gloss_weights, **discriminator_weights}
        )

        self._AA.generate_sample = self.generate_sample
        self._AA.get_variable = self.get_variable
        self._AA.inputs_shape = self.inputs_shape
        self._AA.latent_dim = self.latent_dim

        print(self._AA.summary())

    def compile(
            self,
            adversarial_losses,
            adversarial_weights,
            **kwargs
    ):
        self.adversarial_losses=adversarial_losses
        self.adversarial_weights=adversarial_weights
        autoencoder.compile(
            self,
            **kwargs
        )

    def discriminators_compile(self, **kwargs):
        for k, model in self.adversarial_models.items():
            model['variable'].compile(
                optimizer=self.optimizer,
                loss=self.adversarial_losses[k+'_outputs']()
            )

            print(model['variable'].summary())

