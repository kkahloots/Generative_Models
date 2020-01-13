
import tensorflow as tf

from graphs.basics.AE_graph import make_ae, encode
from training.traditional.autoencoders.AE import AE as basicAE


class AE(basicAE):
    def __init__(
            self,
            model_name,
            inputs_shape,
            outputs_shape,
            latent_dim,
            variables_params,
            restore=None
    ):
        AE.__init__(self,
            model_name=model_name,
            inputs_shape=inputs_shape,
            outputs_shape=outputs_shape,
            latent_dim=latent_dim,
            variables_params=variables_params,
            restore=restore,
            make_ae=make_ae)

        self.encode_graph = encode

    @tf.function
    def feedforward(self, inputs):
        X = inputs[0]
        y = inputs[1]
        z = self.encode(X)
        x_logit = self.decode(tf.concat[z, y])
        return {'x_logit': x_logit, 'latent': z}

    def train_step(self, inputs, names):
        X = inputs[names[0]]
        Xt = inputs[names[1]]
        y = inputs[names[2]]
        with tf.GradientTape() as tape:
            losses_dict = self.loss_functions()
            for loss_name, loss_func in losses_dict.items():
                losses_dict[loss_name] = loss_func(inputs=Xt, predictions=self.feedforward([X,y]))

            losses = -sum([*losses_dict.values()])
        gradients = tape.gradient(losses, self.get_trainables([*self.get_variables().values()]))
        self.optimizer.apply_gradients(zip(gradients, self.get_trainables([*self.get_variables().values()])))
        return losses

    def evaluate_step(self, inputs, names):
        X = inputs[names[0]]
        Xt = inputs[names[1]]
        y = inputs[names[2]]
        losses_dict = self.loss_functions()
        for loss_name, loss_func in losses_dict.items():
            losses_dict[loss_name] = loss_func(inputs=Xt, predictions=self.feedforward([X,y]))
        return losses_dict
