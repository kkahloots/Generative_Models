import logging

import tensorflow as tf
import sys


from graphs.builder import create_models, load_models
from statistical.ae_losses import expected_loglikelihood_with_lower_bound
from utils.reporting.logging import log_message


def create_graph(name, variables_params, restore=None):
    variables_names = [variables['name'] for variables in variables_params]  # ['inference',  'generative']
    variables = create_variables(variables_params=variables_params, model_name=name, restore=restore)

    def get_variables():
        return dict(zip(variables_names, variables))
    return get_variables

def create_losses():
    return {'x_logits': cross_entropy}

def create_trans_losses(input_fn=None, output_fn=None):
    def cross_entropy_fn(x_true, x_logits):
        if input_fn:
            x_true = input_fn(x_true)
        if input_fn:
            x_logits = output_fn(x_true)
        return cross_entropy_fn(x_true, x_logits)

    return {'x_logits': cross_entropy_fn}

def cross_entropy(inputs, x_logits):
    reconstruction_loss = expected_loglikelihood_with_lower_bound(x_true=inputs, x_logits=x_logits)
    Px_xreconst = tf.reduce_mean(-reconstruction_loss)
    return -Px_xreconst

def create_variables(variables_params, model_name, restore=None):
    variables_names = [variables['name'] for variables in variables_params]
    variables = None
    if restore:
        try:
            variables = load_models(restore, [model_name + '_' + var for var in variables_names])

        except Exception as e:
            print(str(e))
            print()
            log_message('Faild to restore old models !', logging.ERROR)

    variables = variables or create_models(variables_params)

    return variables

def encode_fn(**kwargs):
    model = kwargs['model']
    if 'inference' in kwargs['inference_inputs']:
        inputs = kwargs['inference_inputs']['inference']
    else:
        inputs = kwargs['inference_inputs']

    z = model('inference', [inputs])
    return {
        'z_latents': z
    }

def decode_fn(model, latents, output_shape, apply_sigmoid=False):
    x_logits = model('generative', [latents['generative_inputs']])
    if apply_sigmoid:
        probs = tf.sigmoid(x_logits)
        return tf.reshape(tensor=probs, shape=[-1] + [*output_shape], name='x_probablities')
    return tf.reshape(tensor=x_logits, shape=[-1] + [*output_shape], name='x_logits')

def generate_sample(model, input_shape, latents_shape, epsilon=None):
    if epsilon is None:
        epsilon = tf.random.normal(shape=latents_shape)
    generated = decode_fn(model=model, latents={'generative_inputs': epsilon}, output_shape=input_shape, apply_sigmoid=True)
    return generated