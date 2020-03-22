
import tensorflow as tf

from graphs.basics.AE_graph import create_graph, encode_fn
from training.autoencoding_basic.autoencoders.autoencoder import autoencoder as basicAE


class autoencoder(basicAE):
    def __init__(
            self,
            name,
            inputs_shape,
            outputs_shape,
            latent_dim,
            variables_params,
            filepath=None
    ):
        basicAE.__init__(self,
                         name=name,
                         inputs_shape=inputs_shape,
                         outputs_shape=outputs_shape,
                         latent_dim=latent_dim,
                         variables_params=variables_params,
                         filepath=filepath,
                         model_fn=create_graph)

        self.encode_graph = encode_fn

    @tf.function
    def feedforwad(self, inputs):
        X = inputs[0]
        y = inputs[1]
        z = self.encode(X)
        x_logit = self.decode(tf.concat[z, y])
        return {'x_logit': x_logit, 'latent': z}

    def train_step(self, inputs, names):
        try:
            X = inputs[names[0]]
        except:
            X = inputs[0]
        try:
            Xt = inputs[names[1]]
        except:
            Xt = inputs[1]

        try:
            y = inputs[names[2]]
        except:
            y = inputs[2]

        with tf.GradientTape() as tape:
            losses_dict = self.loss_functions()
            for loss_name, loss_func in losses_dict.items():
                losses_dict[loss_name] = loss_func(inputs=Xt, predictions=self.feedforwad([X, y]))

            losses = -sum([*losses_dict.values()])
        gradients = tape.gradient(losses, self.get_trainables([*self.get_variables().values()]))
        self.optimizer.apply_gradients(zip(gradients, self.get_trainables([*self.get_variables().values()])))
        return losses

    def evaluate_step(self, inputs, names):
        try:
            X = inputs[names[0]]
        except:
            X = inputs[0]
        try:
            Xt = inputs[names[1]]
        except:
            Xt = inputs[1]

        try:
            y = inputs[names[2]]
        except:
            y = inputs[2]

        losses_dict = self.loss_functions()
        for loss_name, loss_func in losses_dict.items():
            losses_dict[loss_name] = loss_func(inputs=Xt, predictions=self.feedforwad([X, y]))
        return losses_dict
