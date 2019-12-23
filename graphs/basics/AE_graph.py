import os
import tensorflow as tf
from stats.losses import reconstuction_loss
from stats.pdfs import log_normal_pdf
import logging
from utils.reporting.logging import log_message

from graphs.builder import make_models, load_models, save_models
def make_ae(model_name, input_shape, latent_dim, variables_params, restore=None):
    strategy = make_strategy()
    with strategy.scope():
        variables_names = ['encoder',  'decoder']
        variables = [None for _ in range(len(variables_names))]
        if restore is None:
            variables = make_models(variables_params)
        else:
            variables = load_models(restore, [model_name+'_'+var for var in variables_names])

        def get_variables():
            return dict(zip(variables_names, variables))

        def run_variable(var_name, param):
            return get_variables()[var_name](*param)

        def encode(x):
            z = run_variable('encoder', [x])
            return z

        def decode(z, apply_sigmoid=False):
            logits = run_variable('decoder', [z])
            if apply_sigmoid:
                probs = tf.sigmoid(logits)
                return tf.reshape(probs, shape=[-1] + [*input_shape])
            return tf.reshape(logits, shape=[-1] + [*input_shape])

        def compute_losses(func):
            @tf.function
            def compute_logpx_z(x):
                x, x_logit, z = func(x)
                reconstruction_loss = reconstuction_loss(reconstructed_x=x_logit, true_x=x)
                logpx_z = tf.reduce_mean(-reconstruction_loss)
                return logpx_z

            return {'logpx_z': compute_logpx_z}

        @tf.function
        def generate_sample(eps=None):
            if eps is None:
                eps = tf.random.normal(shape=(100, latent_dim))
            generated = decode(z=eps, apply_sigmoid=True)
            return generated

    return strategy, get_variables, compute_losses, encode, decode, generate_sample,  save_models, load_models


def make_strategy():
    strategy = None
    try:
        tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
        log_message('TPU Strategy ... ', logging.DEBUG)

    except:
        try:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
                tf.distribute.experimental.CollectiveCommunication.NCCL)
            log_message('MultiWorker Mirrored Strategy ... ', logging.DEBUG)
        except:
            strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
            log_message('Mirrored Strategy ... ', logging.DEBUG)

    if strategy is None:
        log_message('Cannot make any strategy for training ... ', logging.ERROR)
        raise NameError('Cannot make any strategy for training ... ')
    return strategy
