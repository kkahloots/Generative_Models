import tensorflow as tf

from graphs.builder import make_models, load_models
from stats.losses import reconstuction_loss
from stats.pdfs import log_normal_pdf
#from graphs.basics.AE_graph import decode

def make_vae(model_name, variables_params, restore=None):
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
    variables = [None for _ in range(len(variables_names))]
    if restore is None:
        variables = make_models(variables_params)
    else:
        variables = load_models(restore, [model_name +'_' + var for var in variables_names])
    return variables

def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

def encode(model, inputs):
    mean, logvar = model('encoder_mean', [inputs]), model('encoder_logvar', [inputs])
    z = reparameterize(mean, logvar)
    return z, mean, logvar



