import tensorflow as tf

from graphs.builder import create_models, load_models
from statistics.ae_losses import reconstuction_loss

def create_graph(name, variables_params, restore=None):
    variables_names = [variables['name'] for variables in variables_params]  # ['inference',  'generative']
    variables = create_variables(variables_params=variables_params, model_name=name, restore=restore)

    def get_variables():
        return dict(zip(variables_names, variables))
    return get_variables


def create_losses():
    return dict(zip(['x_logits'], [bce]))

def bce(inputs, x_logits):
    reconstruction_loss = reconstuction_loss(x_true=inputs, x_logits=x_logits)
    Px_xreconst = tf.reduce_mean(-reconstruction_loss)
    return -Px_xreconst

def create_variables(variables_params, model_name, restore=None):
    variables_names = [variables['name'] for variables in variables_params]
    if restore is None:
        variables = create_models(variables_params)
    else:
        variables = load_models(restore, [model_name + '_' + var for var in variables_names])
    return variables

def encode_fn(**kwargs):
    model = kwargs['model']
    inputs = kwargs['inputs']
    z = model('inference', [inputs])
    return {
        'z_latent': z
    }

def decode_fn(model, latent, inputs_shape, apply_sigmoid=False):
    x_logits = model('generative', [latent])
    if apply_sigmoid:
        probs = tf.sigmoid(x_logits)
        return tf.reshape(tensor=probs, shape=[-1] + [*inputs_shape], name='x_probablities')
    return tf.reshape(tensor=x_logits, shape=[-1] + [*inputs_shape], name='x_logits')

def generate_sample(model, inputs_shape, latent_shape, eps=None):
    if eps is None:
        eps = tf.random.normal(shape=latent_shape)
    generated = decode_fn(model=model, latent=eps, inputs_shape=inputs_shape, apply_sigmoid=True)
    return generated