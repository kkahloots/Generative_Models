import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterable

import tensorflow as tf
from IPython import display
from keras_radam.training import RAdamOptimizer
from tqdm import tqdm

from graphs.basics.AE_graph import make_ae, encode, decode, generate_sample
from graphs.builder import load_models, save_models
from training.callbacks import EarlyStopping
from utils.data_and_files.file_utils import create_if_not_exist, log, inspect_log
from utils.reporting.logging import log_message
from utils.reporting.ploting import plot_and_save_generated


class AE():
    def __init__(
            self,
            model_name,
            inputs_shape,
            outputs_shape,
            latent_dim,
            variables_params,
            restore=None,
            make_ae=make_ae
    ):
        self.make_ae = make_ae
        get_variables, loss_functions= \
            self.make_ae(
                model_name=model_name,
                 variables_params=variables_params,
                restore=restore
            )
        self.inputs_shape = inputs_shape
        self.outputs_shape = outputs_shape
        self.latent_dim = latent_dim
        self.model_name = model_name
        self.get_variables = get_variables
        self.encode_graph = encode
        self.decode_graph = decode
        self.generate_sample = generate_sample
        self.loss_functions = loss_functions
        self.save_models = save_models
        self.load_models = load_models

    def encode(self, inputs, instance_scale=255):
        if inputs.shape == self.inputs_shape:
            inputs = tf.reshape(inputs, (1, ) + self.inputs_shape)
        inputs = tf.cast(inputs, tf.float32)/instance_scale
        return self.encode_graph(model=self.get_varibale, inputs=inputs)

    def decode(self, latent):
        return self.decode_graph(model=self.get_varibale, latent=latent, inputs_shape=self.inputs_shape)

    def get_varibale(self, var_name, param):
        return self.get_variables()[var_name](*param)

    def save_status(self):
        file_Name = os.path.join(self.var_save_dir, self.model_name)
        self.save_models(file_Name, self.get_variables())

    #@tf.function
    def feedforward(self, inputs):
        z = self.encode_graph(model=self.get_varibale, inputs=inputs)
        x_logit = self.decode_graph(model=self.get_varibale, latent=z, inputs_shape=self.inputs_shape)
        return {'x_logit': x_logit, 'latent': z}

    def fit(self, train_dataset, test_dataset,
            instance_name='image',
            instance_scale=255.0,
            epochs=10,
            learning_rate=1e-3,
            random_latent=None,
            recoding_dir='./recoding',
            gray_plot=True,
            generate_epoch=5,
            save_epoch=5
            ):
        assert isinstance(train_dataset, Iterable), 'dataset must be iterable'
        assert isinstance(test_dataset, Iterable), 'dataset must be iterable'

        self.dir_setup(recoding_dir)

        # generate random latent
        latent_shape = [50, self.latent_dim]
        if random_latent is None:
            random_latent = tf.random.normal(shape=latent_shape)
        generated = self.generate_sample(model=self.get_varibale, inputs_shape=self.inputs_shape, latent_shape=latent_shape, eps=random_latent)
        plot_and_save_generated(generated=generated, epoch=0, path=self.image_gen_dir, gray=gray_plot)

        def get_trainables(var_list):
            vars = []
            for var in var_list:
                vars += var.trainable_variables
            return vars

        def train_step(train_x, train_xt=None):
            with tf.GradientTape() as tape:
                losses_dict = self.loss_functions()
                for loss_name, loss_func in losses_dict.items():
                    if train_xt is None:
                        losses_dict[loss_name] = loss_func(inputs=train_x, predictions=self.feedforward(train_x))
                    else:
                        losses_dict[loss_name] = loss_func(inputs=train_xt, predictions=self.feedforward(train_x))

                losses = -sum([*losses_dict.values()])
            gradients = tape.gradient(losses, get_trainables([*self.get_variables().values()]))
            self.optimizer.apply_gradients(zip(gradients, get_trainables([*self.get_variables().values()])))
            return losses

        def evaluate_step(dataset_inputs):
            losses_dict = self.loss_functions()
            for loss_name, loss_func in losses_dict.items():
                losses_dict[loss_name] = loss_func(inputs=dataset_inputs, predictions=self.feedforward(dataset_inputs))
            return losses_dict

        self.optimizer = RAdamOptimizer(learning_rate)

        file_Name = os.path.join(self.csv_log_dir, 'TRAIN_' + self.model_name+'.csv')
        start_epoch = inspect_log(file_Name)

        early_stopper = EarlyStopping(name='on-Test dataset ELBO monitor', patience=5, min_delta=1e-6)
        epochs_pbar = tqdm(iterable=range(start_epoch, start_epoch+epochs), position=0, desc='Epochs Progress')
        for epoch in epochs_pbar:
            # training dataset
            tr_start_time = time.time()
            loss_tr = defaultdict()
            loss_tr['Epoch'] = epoch
            log_message('Training ... ', logging.INFO)
            for i, data_train in enumerate(train_dataset):
                if instance_name=='episode':
                    train_x = tf.cast(data_train[0], dtype=tf.float32)/instance_scale
                else:
                    train_x = tf.cast(data_train[instance_name], dtype=tf.float32) / instance_scale
                total_loss = train_step(train_x)
                tr_losses = evaluate_step(train_x)
                for loss_name, loss_value in tr_losses.items():
                    try:
                        loss_tr[loss_name] += loss_value.numpy()
                    except:
                        loss_tr[loss_name] = loss_value.numpy()
                loss_tr['Total'] = sum([*tr_losses.values()]).numpy()
                epochs_pbar.set_description('Epochs Progress, Training Iterations {}'.format(i))
            tr_end_time = time.time()
            loss_tr['Elapsed'] = '{:06f}'.format(tr_end_time - tr_start_time)

            # testing dataset
            val_start_time = time.time()
            loss_val = defaultdict()
            loss_val['Epoch'] = epoch

            log_message('Testing ... ', logging.INFO)
            tbar = tqdm(iterable=range(100), position=0, desc='Testing ...')
            for i, data_test in enumerate(test_dataset):
                if instance_name == 'episode':
                    test_x = tf.cast(data_test[0], dtype=tf.float32) / instance_scale
                else:
                    test_x = tf.cast(data_test[instance_name], dtype=tf.float32) / instance_scale

                val_losses = evaluate_step(test_x)
                for loss_name, loss_value in val_losses.items():
                    try:
                        loss_val[loss_name] += loss_value.numpy()
                    except:
                        loss_val[loss_name] = loss_value.numpy()
                loss_val['Total'] = sum([*val_losses.values()]).numpy()
                montiored_loss = loss_val['Total']
                tbar.update(i%100)
            val_end_time = time.time()
            loss_val['Elapsed'] = '{:06f}'.format(val_end_time - val_start_time)

            display.clear_output(wait=False)
            log_message("==================================================================", logging.INFO)
            file_Name = os.path.join(self.csv_log_dir, 'TRAIN_' + self.model_name)
            log(file_name=file_Name, message=dict(loss_tr), printed=True)
            log_message("==================================================================", logging.INFO)

            log_message("==================================================================", logging.INFO)
            file_Name = os.path.join(self.csv_log_dir, 'TEST_' + self.model_name)
            log(file_name=file_Name, message=dict(loss_val), printed=True)
            log_message("==================================================================", logging.INFO)

            if epoch%generate_epoch==0:
                generated = self.generate_sample(model=self.get_varibale, inputs_shape=self.inputs_shape, latent_shape=latent_shape, eps=random_latent)
            plot_and_save_generated(generated=generated, epoch=epoch, path=self.image_gen_dir,
                                    gray=gray_plot, save=epoch%generate_epoch==0)

            if epoch%save_epoch == 0:
                log_message('Saving Status in Epoch {}'.format(epoch), logging.CRITICAL)
                self.save_status()

            # Early stopping
            if (early_stopper.stop(montiored_loss)):
                log_message('Aborting Training after {} epoch because no progress ... '.format(epoch), logging.WARN)
                break

    def dir_setup(self, recoding_dir):
        self.recoding_dir = recoding_dir
        self.csv_log_dir = os.path.join(self.recoding_dir, 'csv_log_dir')
        self.image_gen_dir = os.path.join(self.recoding_dir, 'image_gen_dir')
        self.var_save_dir = os.path.join(self.recoding_dir, 'var_save_dir')

        create_if_not_exist([self.recoding_dir, self.csv_log_dir, self.image_gen_dir, self.var_save_dir])

