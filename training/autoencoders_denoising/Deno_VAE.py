import os
import time

from collections import defaultdict
from collections.abc import Iterable
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from IPython import display
import logging
from utils.reporting.logging import log_message
from training.callbacks import EarlyStopping

from keras_radam.training import RAdamOptimizer
from utils.reporting.ploting import plot_and_save_generated
from utils.data_and_files.file_utils import create_if_not_exist, log, inspect_log
from utils.reporting.ploting import animate

from graphs.denoising.Cond_VAE_graph import make_cond_vae
class DenoVAE():
    def __init__(
            self,
            model_name,
            input_shape,
            latent_dim,
            variables_params,
            restore=None
    ):

        strategy, get_variables, compute_losses, \
        encode, decode, generate_sample, save_models, load_models = \
            make_cond_vae(
                model_name=model_name,
                input_shape=input_shape,
                latent_dim=latent_dim,
                variables_params=variables_params,
                restore=restore
            )
        self.model_name = model_name
        self.strategy = strategy
        self.get_variables = get_variables
        self.encode = encode
        self.decode = decode
        self.generate_sample = generate_sample
        self.compute_losses = compute_losses
        self.save_models = save_models
        self.load_models = load_models

    def save_status(self):
        file_Name = os.path.join(self.var_save_dir, self.model_name)
        self.save_models(file_Name, self.get_variables())

    def fit(self, train_dataset, test_dataset,
            instance_name='image',
            instance_scale=255.0,
            epochs=10,
            learning_rate=1e-3,
            random_latent=None,
            latent_dim=10,
            recoding_dir='./recoding',
            gray_plot=True,
            generate_epoch=5,
            save_epoch=5
            ):
        assert isinstance(train_dataset, Iterable), 'dataset must be iterable'
        assert isinstance(test_dataset, Iterable), 'dataset must be iterable'

        self.dir_setup(recoding_dir)

        # generate random latent
        latent_shape = [50, latent_dim]
        if random_latent is None:
            random_latent = tf.random.normal(shape=latent_shape)
        generated = self.generate_sample(latent_shape=latent_shape, eps=random_latent)
        plot_and_save_generated(generated=generated, epoch=0, path=self.image_gen_dir, gray=gray_plot)

        with self.strategy.scope():
            def get_trainables(var_list):
                vars = []
                for var in var_list:
                    vars += var.trainable_variables
                return vars

            train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
            test_dist_dataset = self.strategy.experimental_distribute_dataset(test_dataset)

            @tf.function
            def feedforward(x):
                z, mean, logvar = self.encode(x)
                x_logit = self.decode(z)
                return x, x_logit, z, mean, logvar

            @tf.function
            def distributed_train_epoch(dataset):
                total_loss = 0.0
                num_batches = 0
                for x in dataset:
                    per_replica_losses = self.strategy.experimental_run_v2(train_step,
                                                                      args=(x,))
                    total_loss += self.strategy.reduce(
                        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                    num_batches += 1
                return total_loss / tf.cast(num_batches, dtype=tf.float32)

            def train_step(train_x):
                with tf.GradientTape() as tape:
                    losses_dict = self.compute_losses(feedforward)
                    for loss_name, loss_func in losses_dict.items():
                        losses_dict[loss_name] = loss_func(train_x)
                    losses = -sum([*losses_dict.values()])
                gradients = tape.gradient(losses, get_trainables([*self.get_variables().values()]))
                self.optimizer.apply_gradients(zip(gradients, get_trainables([*self.get_variables().values()])))
                return losses

            @tf.function
            def distributed_train_step(ds):
                per_replica_losses = self.strategy.experimental_run_v2(train_step,
                                                                  args=(ds, ))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            @tf.function
            def distributed_evaluate_step(ds):
                losses_dict = self.compute_losses(feedforward)
                for loss_name, loss_func in losses_dict.items():
                    losses_dict[loss_name] = self.strategy.experimental_run_v2(loss_func,
                                                                        args=(ds, ))

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
                for i, data_train in enumerate(train_dist_dataset):
                    data_train = tf.cast(data_train, dtype=tf.float32)/instance_scale

                    total_loss = distributed_train_step(data_train)
                    tr_losses = distributed_evaluate_step(data_train)
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
                for i, data_test in enumerate(test_dist_dataset):
                    data_test = tf.cast(data_test, dtype=tf.float32) / instance_scale

                    val_losses = distributed_evaluate_step(data_test)
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
                    generated = self.generate_sample(latent_shape=latent_shape, eps=random_latent)
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
