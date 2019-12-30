import os
import tensorflow as tf
from stats.losses import reconstuction_loss
from stats.pdfs import log_normal_pdf
import logging
from utils.reporting.logging import log_message

from graphs.builder import make_models, load_models, save_models
def make_deno_vae(model_name, input_shape, latent_dim, variables_params, restore=None):
    strategy = make_strategy()
    with strategy.scope():
        variables_names = ['encoder_mean', 'encoder_logvar', 'decoder']
        variables = [None for _ in range(len(variables_names))]
        if restore is None:
            variables = make_models(variables_params)
        else:
            variables = load_models(restore, [model_name+'_'+var for var in variables_names])

        def get_variables():
            return dict(zip(variables_names, variables))

        def run_variable(var_name, param):
            return get_variables()[var_name](*param)

        def reparameterize(mean, logvar):
            eps = tf.random.normal(shape=mean.shape)
            return eps * tf.exp(logvar * .5) + mean

        def encode(x):
            mean, logvar = run_variable('encoder_mean', [x]), run_variable('encoder_logvar', [x])
            z = reparameterize(mean, logvar)
            return z, mean, logvar

        def decode(z, apply_sigmoid=False):
            logits = run_variable('decoder', [z])
            if apply_sigmoid:
                probs = tf.sigmoid(logits)
                return tf.reshape(probs, shape=[-1] + [*input_shape])
            return tf.reshape(logits, shape=[-1] + [*input_shape])

        def compute_losses(func):
            @tf.function
            def compute_logpx_z(x):
                xt, xgt = x[0], x[1]
                xt, x_logit, z, mean, logvar = func(xt)
                reconstruction_loss = reconstuction_loss(pred_x=x_logit, true_x=xgt)
                logpx_z = tf.reduce_mean(-reconstruction_loss)
                return logpx_z

            @tf.function
            def compute_logpz(x):
                xt, xgt = x[0], x[1]
                xt, x_logit, z, mean, logvar = func(xt)
                logpz = tf.reduce_mean(log_normal_pdf(z, 0., 0.))
                return logpz

            @tf.function
            def compute_logqz_x(x):
                xt, xgt = x[0], x[1]
                xt, x_logit, z, mean, logvar = func(xt)
                logqz_x = tf.reduce_mean(-log_normal_pdf(z, mean, logvar))
                return logqz_x

            return {'logpx_z': compute_logpx_z, 'logpz': compute_logpz, 'logqz_x': compute_logqz_x}

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
