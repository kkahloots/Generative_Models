
import tensorflow as tf

from graphs.basics.AE_graph import create_graph, encode_fn
from training.traditional.autoencoders.autoencoder import autoencoder as basicAE

class autoencoder(basicAE):
    def __init__(
            self,
            name,
            inputs_shape,
            outputs_shape,
            latent_dim,
            variables_params,
            filepath=None,
            model_fn=create_graph
    ):

        basicAE.__init__(self,
                         name=name,
                         inputs_shape=inputs_shape,
                         outputs_shape=outputs_shape,
                         latent_dim=latent_dim,
                         variables_params=variables_params,
                         filepath=filepath,
                         model_fn=model_fn)

        self.encode_graph = encode_fn

    def train_step(self, inputs, names):
        try:
            X = inputs[names[0]]
        except:
            X = inputs[0]
        try:
            Xt = inputs[names[1]]
        except:
            Xt = inputs[1]
        with tf.GradientTape() as tape:
            losses_dict = self.loss_functions()
            for loss_name, loss_func in losses_dict.items():
                losses_dict[loss_name] = loss_func(inputs=Xt, predictions=self.feedforwad(X))

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
        losses_dict = self.loss_functions()
        for loss_name, loss_func in losses_dict.items():
            losses_dict[loss_name] = loss_func(inputs=Xt, predictions=self.feedforwad(X))
        return losses_dict

