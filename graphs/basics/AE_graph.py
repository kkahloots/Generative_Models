import logging
import os

import tensorflow as tf

from graphs.builder import make_models, load_models, save_models
from stats.losses import reconstuction_loss
from utils.reporting.logging import log_message
from evaluation import quantitive_metrics as qm


def make_ae(model_name, variables_params, restore=None):
    variables_names = [variables['name'] for variables in variables_params]  # ['inference',  'generative']
    variables = make_variables(variables_params=variables_params, model_name=model_name, restore=restore)

    def get_variables():
        return dict(zip(variables_names, variables))

    def loss_functions():
        return dict(zip(['binary_crossentropy'], [compute_Px_xreconst]))

    def frame_quality_metrics():
        return {'ssmi': ssmi, 'psnr': psnr, 'sharp_diff': sharp_diff}

    return get_variables, loss_functions


@tf.function
def compute_Px_xreconst(inputs, predictions):
    x_logit = predictions['x_logit']
    reconstruction_loss = reconstuction_loss(true_x=inputs, pred_x=x_logit)
    Px_xreconst = tf.reduce_mean(-reconstruction_loss)
    return Px_xreconst


@tf.function
def ssmi(inputs, predictions):
    # evaluate difference on each picture separately and combine the results
    episode_len = inputs.shape[0]
    result = 0
    for i in range(episode_len):
        result += qm.ssmi(inputs[i], predictions[i])
    return result


@tf.function
def psnr(inputs, predictions):
    qm.psnr_error(inputs, predictions)


@tf.function
def sharp_diff(inputs, predictions):
    qm.sharp_diff_error(inputs, predictions)


def make_variables(variables_params, model_name, restore=None):
    variables_names = [variables['name'] for variables in variables_params]
    # variables = [None for _ in range(len(variables_names))]
    if restore is None:
        variables = make_models(variables_params)
    else:
        variables = load_models(restore, [model_name + '_' + var for var in variables_names])
    return variables


def encode(model, inputs):
    z = model('inference', [inputs])
    return z


def decode(model, latent, inputs_shape, apply_sigmoid=False):
    logits = model('generative', [latent])
    if apply_sigmoid:
        probs = tf.sigmoid(logits)
        return tf.reshape(probs, shape=[-1] + [*inputs_shape])
    return tf.reshape(logits, shape=[-1] + [*inputs_shape])


@tf.function
def generate_sample(model, inputs_shape, latent_shape, eps=None):
    if eps is None:
        eps = tf.random.normal(shape=latent_shape)
    generated = decode(model=model, latent=eps, inputs_shape=inputs_shape, apply_sigmoid=True)
    return generated


def make_strategy():
    # TODO Remove this promptly not needed i think
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
