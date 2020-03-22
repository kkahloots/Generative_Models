import tensorflow as tf
from graphs.basics.AE_graph import create_variables
from statistics.ae_losses import reconstuction_loss
from statistics.pdfs import log_normal_pdf

# Graph
def create_graph(name, variables_params, restore=None):
    variables_names = [variables['name'] for variables in variables_params] #['inference_mean',  'inference_logvariance', 'generative']
    variables = create_variables(variables_params=variables_params, model_name=name, restore=restore)

    def get_variables():
        return dict(zip(variables_names, variables))
    return get_variables

def reparameterize(mean, logvariance, latent_shape):
    eps = tf.random.normal(shape=latent_shape)
    return tf.add(x=eps * tf.exp(logvariance * .5) , y=mean, name='z_latent')

def encode_fn(**kwargs):
    model = kwargs['model']
    inputs = kwargs['inputs']
    latent_shape = kwargs['latent_shape']
    mean, logvariance = model('inference_mean', [inputs['x_mean']]), model('inference_logvariance', [inputs['x_logvariance']])
    z = reparameterize(mean, logvariance, latent_shape)
    return {
        'z_latent': z,
        'inference_mean': mean,
        'inference_logvariance': logvariance
    }

# losses
def create_losses():
    return {
        'x_logits': logpx_z_fn,
        'z_latent': logpz_fn,
        'x_log_pdf': logqz_x_fn,

    }

def logpx_z_fn(inputs, x_logits):
    reconstruction_loss = reconstuction_loss(x_logits=x_logits, x_true=inputs)
    logpx_z = tf.reduce_mean(-reconstruction_loss)
    return -logpx_z

def logpz_fn(inputs, latent):
    logpz = tf.reduce_mean(log_normal_pdf(latent, 0., 0.))
    return -logpz

def logqz_x_fn(inputs, log_pdf):
    logqz_x = tf.reduce_mean(-log_pdf)
    return -logqz_x
