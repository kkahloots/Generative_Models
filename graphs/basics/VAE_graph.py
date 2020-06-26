import tensorflow as tf
from graphs.basics.AE_graph import create_variables
from statistical.ae_losses import expected_loglikelihood_with_lower_bound
from statistical.pdfs import log_normal_pdf

# Graph
def create_graph(name, variables_params, restore=None):
    variables_names = [variables['name'] for variables in variables_params] #['inference_mean',  'inference_logvariance', 'generative']
    variables = create_variables(variables_params=variables_params, model_name=name, restore=restore)

    def get_variables():
        return dict(zip(variables_names, variables))
    return get_variables

def reparameterize(mean, logvariance, latents_shape):
    epsilon = tf.random.normal(shape=latents_shape)
    return tf.add(x=epsilon * tf.exp(logvariance * .5), y=mean, name='z_latents')

def encode_fn(**kwargs):
    model = kwargs['model']
    inputs = kwargs['inference_inputs']
    latents_shape = kwargs['latents_shape']
    mean, logvariance = model('inference_mean', [inputs['x_mean']]), model('inference_logvariance', [inputs['x_logvariance']])
    z = reparameterize(mean, logvariance, latents_shape)
    return {
        'z_latents': z,
        'inference_mean': mean,
        'inference_logvariance': logvariance
    }

# losses
def create_losses():
    return {
        'x_logits': logpx_z_fn,
        'z_latents': logpz_fn,
        'x_logpdf': logqz_x_fn,

    }

def create_tlosses(log_pdf_fn):
    return {
        'x_logits': logpx_z_fn,
        'z_latents': perpare_tlogpz_fn(log_pdf_fn),
        'x_logpdf': logqz_x_fn,

    }

def logpx_z_fn(inputs, x_logits):
    reconstruction_loss = expected_loglikelihood_with_lower_bound(x_logits=x_logits, x_true=inputs)
    logpx_z = tf.reduce_mean(-reconstruction_loss)
    return -logpx_z

def logpz_fn(inputs, latents):
    logpz = tf.reduce_mean(log_normal_pdf(latents, 0., 0.))
    return -logpz

def perpare_tlogpz_fn(log_pdf_fn):
    def tlogpz_fn(inputs, latents):
        logpz = tf.reduce_mean(log_pdf_fn(latents, 0., 0.))
        return -logpz
    return tlogpz_fn

def logqz_x_fn(inputs, logpdf):
    logqz_x = tf.reduce_mean(-logpdf)
    return -logqz_x
