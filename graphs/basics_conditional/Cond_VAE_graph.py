import tensorflow as tf


from graphs.builder import make_models, load_models
from stats.losses import reconstuction_loss
from stats.pdfs import log_normal_pdf


def make_cond_vae(model_name, variables_params, restore=None):
    variables_names = [variables['name'] for variables in variables_params] #['inference',  'generative']
    variables = make_variables(variables_params=variables_params, model_name=model_name, restore=restore)

    def get_variables():
        return dict(zip(variables_names, variables))

    def loss_functions():
        return {'logpx_z': compute_logpx_z, 'logpz': compute_logpz, 'logqz_x': compute_logqz_x}

    return get_variables, loss_functions

@tf.function
def compute_logpx_z(inputs, predictions):
    x_logit = predictions['x_logit']
    x = inputs
    reconstruction_loss = reconstuction_loss(pred_x=x_logit, true_x=x)
    logpx_z = tf.reduce_mean(-reconstruction_loss)
    return logpx_z

@tf.function
def compute_logpz(inputs, predictions):
    z = predictions['latent']
    logpz = tf.reduce_mean(log_normal_pdf(z, 0., 0.))
    return logpz

@tf.function
def compute_logqz_x(inputs, predictions):
    x_logit, z, mean, logvar = predictions['x_logit'], predictions['latent'], predictions['mean'], predictions['logvar']
    logqz_x = tf.reduce_mean(-log_normal_pdf(z, mean, logvar))
    return logqz_x


def make_variables(variables_params, model_name, restore=None):
    variables_names = [variables['name'] for variables in variables_params]
    #variables = [None for _ in range(len(variables_names))]
    if restore is None:
        variables = make_models(variables_params)
    else:
        variables = load_models(restore, [model_name +'_' + var for var in variables_names])
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

#Previous compute loss
#No diffrence observed between cond and normal except like in line 76
# def compute_losses(func):
#     @tf.function
#     def compute_logpx_z(x):
#         xt, xgt = x[0], x[1]
#         xt, x_logit, z = func(xt)
#         reconstruction_loss = reconstuction_loss(reconstructed_x=x_logit, true_x=xgt)
#         logpx_z = tf.reduce_mean(-reconstruction_loss)
#         return logpx_z
#
#     return {'logpx_z': compute_logpx_z}

