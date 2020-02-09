import tensorflow as tf
from graphs.basics.AE_graph import make_variables
from stats.ae_losses import reconstuction_loss
from stats.pdfs import log_normal_pdf

# Graph
def create_graph(name, variables_params, restore=None):
    variables_names = [variables['name'] for variables in variables_params] #['encoder_mean',  'encoder_logvar', 'generative']
    variables = make_variables(variables_params=variables_params, model_name=name, restore=restore)

    def get_variables():
        return dict(zip(variables_names, variables))
    return get_variables

def reparameterize(mean, logvar, latent_shape):
    eps = tf.random.normal(shape=latent_shape)
    return tf.add(x=eps * tf.exp(logvar * .5) , y=mean, name='x_latent')

def encode_fn(**kwargs):
    model = kwargs['model']
    inputs = kwargs['inputs']
    latent_shape = kwargs['latent_shape']
    mean, logvar = model('encoder_mean', [inputs['x_mean']]), model('encoder_logvar', [inputs['x_logvar']])
    z = reparameterize(mean, logvar, latent_shape)
    return {
        'x_latent': z,
        'x_mean': mean,
        'x_logvar': logvar
    }

# losses
def create_losses():
    return {
        'x_logits': logpx_z_fn,
        'x_latent': logpz_fn,
        'x_log_pdf': logqz_x_fn,

    }

def logpx_z_fn(inputs, x_logits):
    reconstruction_loss = reconstuction_loss(pred_x=x_logits, true_x=inputs)
    logpx_z = tf.reduce_mean(-reconstruction_loss)
    return -logpx_z

def logpz_fn(inputs, latent):
    logpz = tf.reduce_mean(log_normal_pdf(latent, 0., 0.))
    return -logpz

def logqz_x_fn(inputs, log_pdf):
    logqz_x = tf.reduce_mean(-log_pdf)
    return -logqz_x
