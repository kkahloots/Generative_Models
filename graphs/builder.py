import logging
import os

import tensorflow as tf
from tensorflow.keras.models import load_model

from utils.reporting.logging import log_message
from utils.swe.codes import properties, Layer_CD, Activate_CD, Messages_CD, Sampling_CD


def create_layer(layer_cd, lay_dim, kernel_shape=None, addBatchNorm=True, addDropout=True, activate=None):
    assert layer_cd in properties(Layer_CD), Messages_CD.USV.format(layer_cd, 'Layers', properties(Layer_CD))
    assert activate in properties(Activate_CD), Messages_CD.USV.format(activate, 'Activatation', properties(Activate_CD))
    x = []
    if layer_cd in [Layer_CD.Conv, Layer_CD.Deconv]:
        x = [layer_cd(lay_dim, kernel_shape, padding='same')] + x
    else:
        x = [layer_cd(lay_dim)] + x

    if addBatchNorm:
        x = [tf.keras.layers.BatchNormalization()] + x

    if addDropout:
        x = [tf.keras.layers.Dropout(0.2)] + x

    if activate:
        x = [tf.keras.layers.Activation(activate)] + x

    return x

def create_sequence(lay_shapes, isConv=True, kernel_shape=3, sampling_rate=2, addBatchNorm=True, addDropout=True,
                    activate=Activate_CD.relu, last_lay=Sampling_CD.DownSampling):

    assert activate in properties(Activate_CD), Messages_CD.USV.format(activate, 'Activations', properties(Activate_CD))
    assert activate in properties(Activate_CD), Messages_CD.USV.format(activate, 'Activations', properties(Activate_CD))
    x = []
    if isConv:
        lay_cd = Layer_CD.Conv
    else:
        lay_cd = Layer_CD.Dense

    if len(lay_shapes) %2 != 0:
        lay_shapes = lay_shapes[0] + lay_shapes

    for i, lay in enumerate(lay_shapes):
        x = create_layer(lay_cd, lay, kernel_shape=kernel_shape, addBatchNorm=addBatchNorm, addDropout=addDropout, activate=activate) + x

        if isConv:
            if i%2 == 0:
                x = [Sampling_CD.DownSampling((sampling_rate, sampling_rate), padding='same')] + x
            else:
                x = [Sampling_CD.UpSampling((sampling_rate, sampling_rate))] + x

    x = create_layer(lay_cd, lay, kernel_shape=kernel_shape, addBatchNorm=addBatchNorm, addDropout=addDropout, activate=activate) + x
    if last_lay == Sampling_CD.DownSampling:
        x = [Sampling_CD.DownSampling((sampling_rate, sampling_rate), padding='same')] + x
    else:
        x = [Sampling_CD.UpSampling((1, 1), padding='same')] + x

    return x

def create_variable(inputs_shape, outputs_shape, layers=[], name=None):
    variable = \
        tf.keras.Sequential(
        name = name,
        layers=
        [
            tf.keras.layers.Input(shape=inputs_shape, name=name+'_inputs', dtype='float32'),
        ]
        +
            layers
        +
        [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.ActivityRegularization(l1=1e-6, l2=1e-6),
            tf.keras.layers.Activation(None, name=name+'_outputs', dtype='float32')
        ]

    )
    return tf.keras.Model(name=variable.name, inputs=variable.inputs, outputs=variable.outputs)

def save_models(file_name, variables):
    for name, variable in variables.items():
        variable.save(file_name + '_' + name + '.hdf5', overwrite=True)

def load_models(file_name, variables_names):
    log_message('Restore old models ...', logging.DEBUG)
    variables = []
    for name in variables_names:
        variable_path = os.path.join(file_name, name+'.hdf5')
        variable = load_model(variable_path, compile=False)
        variables += [variable]
        log_message(variable.summary(), logging.WARN)
    return variables

def create_models(variables_params):
    vars = []
    for params in variables_params:
        var = create_variable(**params)
        log_message(var.summary(), logging.WARN)
        vars += [var]
    return vars

def run_variable(variable, param):
    return variable(*param)

def layer_stuffing(model):
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Activation):
            if hasattr(layer, 'activation'):
                layer.activation = tf.keras.activations.elu

def clone_model(old_model, new_name, restore=None):
    if restore:
        log_message('Restore old models ...', logging.DEBUG)
        variable_path = os.path.join(restore, new_name+'.hdf5')
        variable = load_model(variable_path, compile=False)

    else:
        temp_layers = tf.keras.models.clone_model(old_model).layers
        temp_layers.append(tf.keras.layers.Flatten())
        temp_layers.append(tf.keras.layers.Dense(units=1, activation='linear', name=new_name+'_outputs'))
        temp_layers = tf.keras.Sequential(temp_layers)
        variable = tf.keras.Model(
            name=new_name,
            inputs=temp_layers.inputs,
            outputs=temp_layers.outputs
        )
    return variable